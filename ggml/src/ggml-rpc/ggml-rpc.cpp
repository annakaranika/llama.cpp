#include "ggml-rpc.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>  // mkdir, for the on-disk weight cache

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <winsock2.h>
#else
#    include <arpa/inet.h>
#    include <netdb.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif
#include <cstring>

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

#define RPC_MAX_DEVICES 16

inline static void ggml_vec_add_f32(const int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

// cross-platform socket
struct socket_t {
    sockfd_t fd;

    socket_t(sockfd_t fd) : fd(fd) {}

    ~socket_t() {
        GGML_PRINT_DEBUG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
        closesocket(this->fd);
#else
        close(this->fd);
#endif
    }
};

// all RPC structures must be packed
#pragma pack(push, 1)

// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];

    char padding[4];
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_SET_SPLIT,
    RPC_CMD_CREATE_PEER_CONNECTION,
    RPC_CMD_ALL_REDUCE,
    RPC_CMD_DO_COMPUTATION,
    RPC_CMD_LOAD_CACHED,       // "do you have this slice cached? if so load it into the buffer" (skip upload)
    RPC_CMD_SET_TENSOR_CACHE,  // set_tensor that ALSO persists the slice to the on-disk weight cache
    RPC_CMD_COUNT,
};

struct rpc_msg_get_alloc_size_req {
    rpc_tensor tensor;
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_split_init_tensor_req {
    rpc_tensor tensor;
    int64_t    row_low;
    int64_t    row_high;
};

struct rpc_msg_alloc_buffer_req {
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t  value;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t   offset;
    uint64_t   size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_graph_compute_rsp {
    uint8_t result;
    // std::vector<uint8_t> output; // output tensors
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_set_split_rsp {
    uint8_t result;
};

struct rpc_msg_create_peer_connection_req {
    uint8_t device_id;                        //id for the current server
    uint8_t device_count;                     //num of servers
    char    endpoints[RPC_MAX_DEVICES][256];  //endpoints to connect to
};

struct rpc_msg_create_peer_connection_rsp {
    uint8_t result;
};

struct rpc_msg_do_computation_req {
    uint8_t graph_number;
};

// Weight cache -- avoid re-uploading identical weights over the network every run.
//   LOAD_CACHED: client asks "do you have the slice with this content hash?". On a
//     hit the server loads it from its on-disk cache straight into the buffer (no
//     upload) and reports hit=1. req carries the dst tensor + the content hash.
//   SET_TENSOR_CACHE (raw payload | rpc_tensor | offset(8) | hash(8) | data |): sent
//     on a miss -- writes the buffer like SET_TENSOR AND persists the slice to the
//     cache file keyed by hash so the next run hits.
struct rpc_msg_load_cached_req {
    rpc_tensor tensor;
    uint64_t   hash;
};

struct rpc_msg_load_cached_rsp {
    uint8_t hit;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = { 0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24,
                              0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03 };
    return &guid;
}

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    std::string name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    void *                    base_ptr;
    uint64_t                  remote_ptr;
};

struct ggml_tensor_extra_rpc {
    struct ggml_backend_rpc_buffer_context * buffer_ctx[RPC_MAX_DEVICES];
    //maybe we don't need to store the rows
    std::pair<int64_t, int64_t>              rows[RPC_MAX_DEVICES];
    int                                      split_dim = -1;
};

//split context
struct ggml_backend_rpc_split_buffer_type_context {
    // int main_device;
    std::string                        endpoint;
    size_t                             alignment;
    size_t                             max_size;
    std::array<float, RPC_MAX_DEVICES> tensor_split;
    std::string                        name;
};

static bool split = false;

struct ggml_backend_rpc_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static ggml_backend_rpc_reg_context * reg_ctx = new ggml_backend_rpc_reg_context;

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    std::string name;
};

static int ggml_backend_rpc_get_device_count() {
    return reg_ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_rpc_get_device(int id) {
    if (id < 0 || id >= ggml_backend_rpc_get_device_count()) {
        return nullptr;
    }
    return reg_ctx->devices[id];
}

static int ggml_backend_rpc_get_device_id(const char * endpoint) {
    for (int i = 0; i < ggml_backend_rpc_get_device_count(); ++i) {
        if (reg_ctx->devices[i]->context &&
            std::strcmp(((ggml_backend_rpc_device_context *) reg_ctx->devices[i]->context)->endpoint.c_str(),
                        endpoint) == 0) {
            return i;
        }
    }
    return -1;
}

// RPC helper functions

static std::shared_ptr<socket_t> make_socket(sockfd_t fd) {
#ifdef _WIN32
    if (fd == INVALID_SOCKET) {
        return nullptr;
    }
#else
    if (fd < 0) {
        return nullptr;
    }
#endif
    return std::make_shared<socket_t>(fd);
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    int ret  = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret  = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *) &flag, sizeof(int));
    return ret == 0;
}

static std::shared_ptr<socket_t> socket_connect(const char * host, int port) {
    struct sockaddr_in addr;
    auto               sockfd   = socket(AF_INET, SOCK_STREAM, 0);
    auto               sock_ptr = make_socket(sockfd);
    if (sock_ptr == nullptr) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        fprintf(stderr, "Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    addr.sin_family         = AF_INET;
    addr.sin_port           = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        fprintf(stderr, "Cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (connect(sock_ptr->fd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
        return nullptr;
    }
    return sock_ptr;
}

static std::shared_ptr<socket_t> socket_accept(sockfd_t srv_sockfd) {
    auto client_socket_fd = accept(srv_sockfd, NULL, NULL);
    auto client_socket    = make_socket(client_socket_fd);
    if (client_socket == nullptr) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        fprintf(stderr, "Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    return client_socket;
}

static std::shared_ptr<socket_t> create_server_socket(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    auto sock   = make_socket(sockfd);
    if (sock == nullptr) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        fprintf(stderr, "Failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        fprintf(stderr, "Invalid host address: %s\n", host);
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family      = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port        = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        return nullptr;
    }
    return sock;
}

static bool send_data(sockfd_t sockfd, const void * data, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        ssize_t n = send(sockfd, (const char *) data + bytes_sent, size - bytes_sent, 0);
        if (n < 0) {
            return false;
        }
        bytes_sent += n;
    }
    return true;
}

static bool recv_data(sockfd_t sockfd, void * data, size_t size) {
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        ssize_t n = recv(sockfd, (char *) data + bytes_recv, size - bytes_recv, 0);
        if (n <= 0) {
            if (n < 0) {
                GGML_LOG_INFO("recv failed: %s\n", strerror(errno));
            } else {
                GGML_LOG_INFO("Connection closed by peer\n");
            }
            return false;
        }
        bytes_recv += n;
    }
    return true;
}

static bool send_msg(sockfd_t sockfd, const void * msg, size_t msg_size) {
    if (!send_data(sockfd, &msg_size, sizeof(msg_size))) {
        return false;
    }
    return send_data(sockfd, msg, msg_size);
}

static bool recv_msg(sockfd_t sockfd, void * msg, size_t msg_size) {
    uint64_t size;
    if (!recv_data(sockfd, &size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return recv_data(sockfd, msg, msg_size);
}

static bool recv_msg(sockfd_t sockfd, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!recv_data(sockfd, &size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        fprintf(stderr, "Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return recv_data(sockfd, input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    port = std::stoi(endpoint.substr(pos + 1));
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(const std::shared_ptr<socket_t> & sock, enum rpc_cmd cmd, const void * input,
                         size_t input_size, void * output, size_t output_size) {
    if (sock == nullptr) {
        GGML_LOG_INFO("[send_rpc_cmd] NULL socket for cmd %d\n", (int) cmd);
        return false;
    }
    uint8_t cmd_byte = cmd;
    if (!send_data(sock->fd, &cmd_byte, sizeof(cmd_byte))) {
        GGML_LOG_INFO("Failed to send command byte %d\n", cmd_byte);
        return false;
    }
    if (!send_data(sock->fd, &input_size, sizeof(input_size))) {
        GGML_LOG_INFO("Failed to send input size %zu\n", input_size);
        return false;
    }
    if (!send_data(sock->fd, input, input_size)) {
        GGML_LOG_INFO("Failed to send input data of size %zu\n", input_size);
        return false;
    }
    // TODO: currently the output_size is always known, do we need support for commands with variable output size?
    // even if we do, we can skip sending output_size from the server for commands with known output size
    uint64_t out_size;
    if (!recv_data(sock->fd, &out_size, sizeof(out_size))) {
        GGML_LOG_INFO("Failed to receive output size\n");
        return false;
    }
    if (out_size != output_size) {
        GGML_LOG_INFO("Expected output size %zu, but got %" PRIu64 "\n", output_size, out_size);
        return false;
    }
    if (!recv_data(sock->fd, output, output_size)) {
        GGML_LOG_INFO("Failed to receive output data of size %" PRIu64 "\n", out_size);
        return false;
    }
    return true;
}

// RPC client-side implementation

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex                                               mutex;
    std::lock_guard<std::mutex>                                     lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;
    static bool                                                     initialized = false;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int         port;
    if (!parse_endpoint(endpoint, host, port)) {
        return nullptr;
    }
#ifdef _WIN32
    if (!initialized) {
        WSADATA wsaData;
        int     res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            return nullptr;
        }
        initialized = true;
    }
#else
    GGML_UNUSED(initialized);
#endif
    auto sock = socket_connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    GGML_LOG_INFO("[%s] connected to %s, sockfd=%d\n", __func__, endpoint.c_str(), sock->fd);
    sockets[endpoint] = sock;
    return sock;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    //how to free a buffer on other servers?
    ggml_backend_rpc_buffer_context * ctx     = (ggml_backend_rpc_buffer_context *) buffer->context;
    rpc_msg_free_buffer_req           request = { ctx->remote_ptr };
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    GGML_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *) buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = { ctx->remote_ptr };
    rpc_msg_buffer_get_base_rsp response;
    bool                        status =
        send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    GGML_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static void * ggml_backend_rpc_buffer_context_get_base(ggml_backend_rpc_buffer_context * ctx) {
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = { ctx->remote_ptr };
    rpc_msg_buffer_get_base_rsp response;
    bool                        status =
        send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    GGML_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    result.id   = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer) {
        ggml_backend_buffer_t             buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx    = (ggml_backend_rpc_buffer_context *) buffer->context;
        result.buffer                            = ctx->remote_ptr;
    } else {
        result.buffer = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

//struct description: spilt buffer needs to have a context. Here we define the context to store the information
struct ggml_backend_rpc_split_buffer_context {
    ~ggml_backend_rpc_split_buffer_context() {
        for (ggml_tensor_extra_rpc * extra : tensor_extras) {
            auto * ctx_item = extra->buffer_ctx;
            for (int i = 0; i < RPC_MAX_DEVICES; ++i) {
                if (ctx_item[i]) {
                    rpc_msg_free_buffer_req request = { ctx_item[i]->remote_ptr };
                    bool                    status =
                        send_rpc_cmd(ctx_item[i]->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
                    GGML_ASSERT(status);
                }
                delete ctx_item[i];
            }
            delete extra;
        }
    }

    std::vector<ggml_tensor_extra_rpc *> tensor_extras;
};

//by rows
static rpc_tensor split_serialize_tensor(const ggml_tensor * tensor, const ggml_tensor_extra_rpc * extra,
                                         int device_id) {
    rpc_tensor result;
    result.id   = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (extra) {
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *) extra->buffer_ctx[device_id];
        result.buffer = ctx->remote_ptr;  //use remote_ptr to find the exact buffer on the remote server
        result.data   = reinterpret_cast<uint64_t>(
            ggml_backend_rpc_buffer_context_get_base(reinterpret_cast<ggml_backend_rpc_buffer_context *>(ctx)));
    } else {
        result.buffer = 0;
        GGML_LOG_INFO("Error: tensor %s has no extra info for split\n", tensor->name);
    }
    int split_dim = extra->split_dim;
    if (split_dim == 1) {
        int row_low  = extra->rows[device_id].first;
        int row_high = extra->rows[device_id].second;

        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            if (i == 1) {
                result.ne[i] = row_high - row_low;  //ne1 is the number of elements in a column, that is, rows
            } else {
                result.ne[i] = tensor->ne[i];
            }
            if (i <= 1) {
                result.nb[i] = tensor->nb[i];  //recalculate nb after ne changes
            } else {
                result.nb[i] = result.nb[i - 1] * result.ne[i - 1];
            }
        }
    } else if (split_dim == 0) {
        int col_low  = extra->rows[device_id].first;
        int col_high = extra->rows[device_id].second;
        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            if (i == 0) {
                result.ne[i] = col_high - col_low;  //ne1 is the number of elements in a column, that is, rows
            } else {
                result.ne[i] = tensor->ne[i];
            }
            if (i == 0) {
                result.nb[i] = tensor->nb[i];  //recalculate nb after ne changes
            } else if (i == 1) {
                result.nb[1] = result.nb[0] * (result.ne[0] / ggml_blck_size(tensor->type));
            } else {
                result.nb[i] = result.nb[i - 1] * result.ne[i - 1];
            }
        }
    } else {
        GGML_LOG_INFO("[%s] split serialize tensor with non-split tensor %s\n", __func__, tensor->name);
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static std::array<float, RPC_MAX_DEVICES> tensor_splits;
static bool                               multi_cpy = true;

static void ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context *      ctx      = (ggml_backend_rpc_buffer_context *) buffer->context;
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *) buffer->buft->context;
    // GGML_LOG_INFO("[%s] initializing tensor %s, type=%d, buffer=%" PRIx64 " data=%" PRIx64 " operation:%d\n", __func__, tensor->name, tensor->type, ctx->remote_ptr, reinterpret_cast<uint64_t>(tensor->data), tensor->op);
    bool                                   cache    = false;
    //if split, store a copy of tensor on every rpc server, and store the buffer context for each server in extra
    if (split && multi_cpy) {
        if (tensor->extra == NULL) {
            ggml_tensor_extra_rpc * extra;
            bool                    found = false;
            if (strncmp(tensor->name, "cache", 5) == 0) {
                // GGML_LOG_INFO("tensor is cache: %s\n",tensor->name);
                cache = true;
                extra = new ggml_tensor_extra_rpc();
            } else if (tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_PERMUTE ||
                       tensor->op == GGML_OP_TRANSPOSE) {
                found = true;
                extra = (ggml_tensor_extra_rpc *) tensor->src[0]->extra;
            } else if (tensor->op == GGML_OP_CPY) {
                found = true;
                extra = (ggml_tensor_extra_rpc *) tensor->src[1]->extra;
            } else {
                extra = new ggml_tensor_extra_rpc();
            }

            //allocate buffer on other servers
            // Allocate each device's buffer concurrently: extra->buffer_ctx[id]/rows[id]
            // are per-id (disjoint) writes, sockets are pre-fetched + held so the weak_ptr
            // cache can't churn under the workers. RPC_SERIAL_UPLOAD=1 reverts to serial.
            // (On alloc failure we set buffer_ctx[id]=nullptr -- which downstream already
            //  null-checks -- instead of `delete extra`, which under threading would
            //  double-free and was a use-after-free even serially via `tensor->extra=extra`.)
            static const bool                      serial_alloc = getenv("RPC_SERIAL_UPLOAD") != nullptr;
            const int                              n_dev        = ggml_backend_rpc_get_device_count();
            std::vector<std::shared_ptr<socket_t>> socks(n_dev);
            for (int id = 0; id < n_dev; ++id) {
                socks[id] = get_socket(((ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context)->endpoint);
            }
            std::vector<std::thread> threads;
            auto                     alloc_one = [&](int id) {
                if (!found) {
                    size_t size;
                    if (cache) {
                        float split_part = (id == ggml_backend_rpc_get_device_count() - 1) ?
                                                                   (1 - tensor_splits[id]) :
                                                                   (tensor_splits[id + 1] - tensor_splits[id]);
                        size = split_part * tensor->ne[0] * tensor->nb[0] / ggml_blck_size(tensor->type);
                        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                            size += (tensor->ne[i] - 1) * tensor->nb[i];
                        }
                    } else {
                        size = ggml_nbytes(tensor);
                    }
                    rpc_msg_alloc_buffer_req          request = { size };  //the size to allocate
                    rpc_msg_alloc_buffer_rsp          response;
                    ggml_backend_rpc_device_context * dev_ctx =
                        (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
                    if (dev_ctx->endpoint ==
                        buft_ctx->endpoint) {  //for the main device, we already have the buffer context
                        extra->buffer_ctx[id] = ctx;
                        extra->rows[id]       = { 0, tensor->ne[0] };
                        return;
                    }
                    bool status = send_rpc_cmd(socks[id], RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response,
                                                                   sizeof(response));
                    GGML_ASSERT(status);
                    if (response.remote_ptr != 0) {
                        extra->buffer_ctx[id] =
                            new ggml_backend_rpc_buffer_context{ socks[id], nullptr, response.remote_ptr };
                        extra->rows[id] = { 0, tensor->ne[0] };
                    } else {
                        GGML_LOG_INFO("[%s] failed to allocate buffer for device %d\n", __func__, id);
                        extra->buffer_ctx[id] = nullptr;
                    }
                } else {
                    extra->rows[id] = { 0, tensor->ne[0] };
                }
            };
            for (int id = 0; id < n_dev; ++id) {
                if (serial_alloc) {
                    alloc_one(id);
                } else {
                    threads.emplace_back(alloc_one, id);
                }
            }
            for (auto & t : threads) {
                if (t.joinable()) {
                    t.join();
                }
            }
            tensor->extra = extra;
        }
    }

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        // if split, we need to send the tensor to all other devices
        if (split && multi_cpy) {
            ggml_tensor_extra_rpc * extra = (ggml_tensor_extra_rpc *) tensor->extra;
            for (int id = 0; id < ggml_backend_rpc_get_device_count(); ++id) {
                ggml_backend_rpc_device_context * dev_ctx =
                    (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
                if (dev_ctx->endpoint == buft_ctx->endpoint) {
                    rpc_msg_init_tensor_req request;
                    request.tensor = serialize_tensor(tensor);
                    if (cache) {
                        float split_part     = (id == ggml_backend_rpc_get_device_count() - 1) ?
                                                   (1 - tensor_splits[id]) :
                                                   (tensor_splits[id + 1] - tensor_splits[id]);
                        request.tensor.ne[0] = split_part * tensor->ne[0];  //TODO: Need Alignment
                        request.tensor.nb[1] =
                            request.tensor.ne[0] * request.tensor.nb[0] / ggml_blck_size(tensor->type);
                        request.tensor.nb[2] = request.tensor.ne[1] * request.tensor.nb[1];
                        request.tensor.nb[3] = request.tensor.ne[2] * request.tensor.nb[2];
                    }
                    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
                    GGML_ASSERT(status);
                    continue;
                }
                rpc_msg_init_tensor_req request;

                //serialize tensor for the specific device
                request.tensor = serialize_tensor(tensor);
                if (cache) {
                    float split_part     = (id == ggml_backend_rpc_get_device_count() - 1) ?
                                               (1 - tensor_splits[id]) :
                                               (tensor_splits[id + 1] - tensor_splits[id]);
                    request.tensor.ne[0] = split_part * tensor->ne[0];
                    request.tensor.nb[1] = request.tensor.ne[0] * request.tensor.nb[0] / ggml_blck_size(tensor->type);
                    request.tensor.nb[2] = request.tensor.ne[1] * request.tensor.nb[1];
                    request.tensor.nb[3] = request.tensor.ne[2] * request.tensor.nb[2];
                }
                //change the buffer and data pointer for this device
                if (extra->buffer_ctx[id] == nullptr) {
                    GGML_LOG_INFO("[%s] buffer context for device %d is null\n", __func__, id);
                } else {
                    if (request.tensor.buffer != extra->buffer_ctx[id]->remote_ptr) {
                        request.tensor.buffer = extra->buffer_ctx[id]->remote_ptr;
                        // GGML_LOG_INFO("init\n");
                        request.tensor.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                            reinterpret_cast<ggml_backend_rpc_buffer_context *>(extra->buffer_ctx[id])));
                    }
                }

                bool status = send_rpc_cmd(get_socket(dev_ctx->endpoint), RPC_CMD_INIT_TENSOR, &request,
                                           sizeof(request), nullptr, 0);
                GGML_ASSERT(status);
            }
        } else {
            rpc_msg_init_tensor_req request;

            request.tensor = serialize_tensor(tensor);

            bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
            GGML_ASSERT(status);
        }
    }
}

// Content hash of a weight slice -- the key for the on-disk weight cache. FNV-1a
// 64-bit: cheap (~GB/s), and a content key means the cache stays correct across
// model swaps (different bytes -> different key -> miss -> re-upload).
static uint64_t rpc_fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t        h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    // GGML_LOG_INFO("[%s] setting tensor %s, offset=%zu, size=%zu\n", __func__, tensor->name, offset, size);
    // GGML_LOG_INFO("ne0 = %ld ne1 = %ld nb0 = %ld nb1 =%ld nb2 = %ld\n",tensor->ne[0],tensor->ne[1],tensor->nb[0],tensor->nb[1],tensor->nb[2]);
    ggml_backend_rpc_buffer_type_context * buft_ctx   = (ggml_backend_rpc_buffer_type_context *) buffer->buft->context;
    ggml_backend_rpc_buffer_context *      ctx        = (ggml_backend_rpc_buffer_context *) buffer->context;
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    size_t                                 input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t>                   input(input_size, 0);
    rpc_tensor                             rpc_tensor1 = serialize_tensor(tensor);
    // GGML_LOG_INFO("[%s] rpc_tensor.data=%" PRIx64 ", rpc_tensor.buffer=%" PRIx64 "\n", __func__, rpc_tensor1.data, rpc_tensor1.buffer);
    memcpy(input.data(), &rpc_tensor1, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR, input.data(), input.size(), nullptr, 0);
    if (!status) {
        GGML_LOG_INFO("[%s] failed to set tensor %s, offset=%zu, size=%zu\n", __func__, tensor->name, offset, size);
    }
    GGML_ASSERT(status);

    // if split, we need to set the tensor on all other devices
    if (split && multi_cpy) {
        ggml_tensor_extra_rpc *                extra         = (ggml_tensor_extra_rpc *) tensor->extra;
        const int                              n_dev         = ggml_backend_rpc_get_device_count();
        // Upload the replicated copy to every other device concurrently. Pre-fetch
        // and HOLD the sockets on the main thread first (the worker threads must
        // not race get_socket / let the weak_ptr-cached sockets churn -- see the
        // split upload). RPC_SERIAL_UPLOAD=1 reverts to one-at-a-time.
        static const bool                      serial_upload = getenv("RPC_SERIAL_UPLOAD") != nullptr;
        std::vector<std::shared_ptr<socket_t>> socks(n_dev);
        for (int id = 0; id < n_dev; ++id) {
            socks[id] = get_socket(((ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context)->endpoint);
        }
        std::vector<std::thread> threads;
        auto                     send_copy = [&](int id) {
            ggml_backend_rpc_device_context * dev_ctx =
                (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
            if (dev_ctx->endpoint == buft_ctx->endpoint) {
                return;
            }
            // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
            size_t               input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
            std::vector<uint8_t> input_(input_size, 0);
            rpc_tensor           rpc_tensor2 = serialize_tensor(tensor);
            //change the buffer and data pointer for this device
            if (extra->buffer_ctx[id] == nullptr) {
                GGML_LOG_INFO("[%s] buffer context for device %d is null\n", __func__, id);
            } else {
                rpc_tensor2.buffer = extra->buffer_ctx[id]->remote_ptr;
                rpc_tensor2.data = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                    reinterpret_cast<ggml_backend_rpc_buffer_context *>(extra->buffer_ctx[id])));
            }
            memcpy(input_.data(), &rpc_tensor2, sizeof(rpc_tensor));
            memcpy(input_.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
            memcpy(input_.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);

            bool status = send_rpc_cmd(socks[id], RPC_CMD_SET_TENSOR, input_.data(), input_.size(), nullptr, 0);
            if (!status) {
                GGML_LOG_INFO("[%s] failed to set tensor %s, offset=%zu, size=%zu\n", __func__, tensor->name, offset,
                                                  size);
            }
            GGML_ASSERT(status);
        };
        for (int id = 0; id < n_dev; ++id) {
            if (serial_upload) {
                send_copy(id);
            } else {
                threads.emplace_back(send_copy, id);
            }
        }
        for (auto & t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                               size_t offset, size_t size) {
    // GGML_LOG_INFO("[%s] getting tensor %s, offset=%zu, size=%zu\n", __func__, tensor->name, offset, size);
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *) buffer->context;
    rpc_msg_get_tensor_req            request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size   = size;
    bool status    = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    GGML_ASSERT(status);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src,
                                               ggml_tensor * dst) {
    bool                              result     = true;
    // check if src and dst are on the same server
    ggml_backend_buffer_t             src_buffer = src->buffer;
    ggml_backend_rpc_buffer_context * src_ctx    = (ggml_backend_rpc_buffer_context *) src_buffer->context;
    ggml_backend_buffer_t             dst_buffer = dst->buffer;
    ggml_backend_rpc_buffer_context * dst_ctx    = (ggml_backend_rpc_buffer_context *) dst_buffer->context;
    if (src_ctx->sock != dst_ctx->sock) {
        return false;
    }
    ggml_backend_rpc_buffer_context *      ctx      = (ggml_backend_rpc_buffer_context *) buffer->context;
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *) buffer->buft->context;
    rpc_msg_copy_tensor_req                request;
    request.src = serialize_tensor(src);
    request.dst = serialize_tensor(dst);
    rpc_msg_copy_tensor_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
    GGML_ASSERT(status);
    result = response.result & result;
    if (split && multi_cpy) {
        ggml_tensor_extra_rpc * src_extra = (ggml_tensor_extra_rpc *) src->extra;
        ggml_tensor_extra_rpc * dst_extra = (ggml_tensor_extra_rpc *) dst->extra;
        for (int id = 0; id < ggml_backend_rpc_get_device_count(); ++id) {
            ggml_backend_rpc_device_context * dev_ctx =
                (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
            if (dev_ctx->endpoint == buft_ctx->endpoint) {
                continue;
            }
            if (src_extra->buffer_ctx[id]->sock != dst_extra->buffer_ctx[id]->sock) {
                result = false;  // src and dst are on different devices, cannot copy
            }
            rpc_msg_copy_tensor_req request;

            //serialize src and dst tensors for the specific device
            request.src = serialize_tensor(src);
            request.dst = serialize_tensor(dst);

            if (src_extra->buffer_ctx[id] == nullptr) {
                GGML_LOG_INFO("[%s] buffer context for device %d is null\n", __func__, id);
            } else {
                request.src.buffer = src_extra->buffer_ctx[id]->remote_ptr;
                // GGML_LOG_INFO("cpy\n");
                request.src.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                    reinterpret_cast<ggml_backend_rpc_buffer_context *>(src_extra->buffer_ctx[id])));
            }

            if (dst_extra->buffer_ctx[id] == nullptr) {
                GGML_LOG_INFO("[%s] buffer context for device %d is null\n", __func__, id);
            } else {
                request.dst.buffer = dst_extra->buffer_ctx[id]->remote_ptr;
                // GGML_LOG_INFO("cpy\n");
                request.dst.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                    reinterpret_cast<ggml_backend_rpc_buffer_context *>(dst_extra->buffer_ctx[id])));
            }

            rpc_msg_copy_tensor_rsp response;
            bool status = send_rpc_cmd(get_socket(dev_ctx->endpoint), RPC_CMD_COPY_TENSOR, &request, sizeof(request),
                                       &response, sizeof(response));
            GGML_ASSERT(status);
            result = response.result & result;
        }
    }
    return result;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    //how to clear the buffer on other servers?
    // GGML_LOG_INFO("[%s] clearing buffer, remote_ptr=%" PRIx64 ", value=%d\n", __func__, ((ggml_backend_rpc_buffer_context *)buffer->context)->remote_ptr, value);
    ggml_backend_rpc_buffer_context * ctx     = (ggml_backend_rpc_buffer_context *) buffer->context;
    rpc_msg_buffer_clear_req          request = { ctx->remote_ptr, value };
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    GGML_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

// The split grid = the largest quant block among the model's split weights,
// discovered at load (rpc_note_split_block, called from rpc_get_*_split as each
// weight is split). Any split of a contraction dimension -- and the F32
// activation that feeds it (which has no block of its own) -- must tile on this
// grid so the producer's and consumer's slices agree. K-quants = 256, legacy
// Q4_0/Q8_0 = 32, F16/F32 = 1. Starts at 1, raised as split weights are seen, so
// it is correct by the time the first forward runs.
static int64_t g_rpc_split_block = 1;

static inline void rpc_note_split_block(const ggml_tensor * tensor) {
    g_rpc_split_block = std::max<int64_t>(g_rpc_split_block, ggml_blck_size(tensor->type));
}

// Canonical per-device split boundary: device `id` gets [*low, *high) of `total`,
// each boundary floored to a multiple of `align` (so a quantized dimension splits
// on whole blocks). Single source of truth for rpc_get_row_split/rpc_get_col_split
// AND the compute-time activation split, so a weight and the activation feeding it
// tile on the exact same grid.
static void rpc_split_range(int64_t total, int64_t align, const std::array<float, RPC_MAX_DEVICES> & tensor_split,
                            int id, int64_t * low, int64_t * high) {
    align              = std::max<int64_t>(align, 1);
    const int64_t devs = ggml_backend_rpc_get_device_count();
    *low               = (id == 0) ? 0 : (int64_t) (total * tensor_split[id]);
    *low -= *low % align;
    *high = (id == devs - 1) ? total : (int64_t) (total * tensor_split[id + 1]);
    *high -= *high % align;
}

static int64_t rpc_split_count(int64_t total, int64_t align, const std::array<float, RPC_MAX_DEVICES> & tensor_split,
                               int id) {
    int64_t low  = 0;
    int64_t high = 0;
    rpc_split_range(total, align, tensor_split, id, &low, &high);
    return high - low;
}

//split buffer interface
static int64_t rpc_get_row_rounding(const std::array<float, RPC_MAX_DEVICES> & tensor_split) {
    int64_t row_rounding = 0;
    for (int id = 0; id < ggml_backend_rpc_get_device_count(); ++id) {
        if (tensor_split[id] >= (id + 1 < ggml_backend_rpc_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
            continue;
        }
        int64_t alignment =
            ggml_backend_buft_get_alignment(reg_ctx->devices[id]->iface.get_buffer_type(reg_ctx->devices[id]));
        row_rounding = std::max(row_rounding, alignment);
    }
    return row_rounding;
}

static void rpc_get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor,
                              const std::array<float, RPC_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    rpc_note_split_block(tensor);  // record this weight's quant block in the split grid
    int64_t       rounding = rpc_get_row_rounding(tensor_split);
    // A row-split tensor whose split dim pairs with a downstream COLUMN split on
    // the same dim (ffn_gate/up rows -> ffn_down cols; q rows -> attn_output cols)
    // must use the SAME boundaries as that col split (block-aligned), else producer
    // and consumer slice the dim differently and the result is garbage. Only
    // block-align when each device still gets >= one quant block, so a sub-block
    // per-head split (k/v: num_kv_heads*d_k = 256, 64/device at N=4) is NOT
    // collapsed onto a single device.
    const int64_t devs     = ggml_backend_rpc_get_device_count();
    const int64_t block    = ggml_blck_size(tensor->type);
    if (devs > 0 && nrows / devs >= block) {
        rounding = std::max(rounding, block);
    }
    rpc_split_range(nrows, rounding, tensor_split, id, row_low, row_high);
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split * ggml_row_size(tensor->type, tensor->ne[0]);
}

static int64_t rpc_get_col_rounding(const std::array<float, RPC_MAX_DEVICES> & tensor_split) {
    int64_t col_rounding = 0;
    for (int id = 0; id < ggml_backend_rpc_get_device_count(); ++id) {
        if (tensor_split[id] >= (id + 1 < ggml_backend_rpc_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
            continue;
        }
        int64_t alignment =
            ggml_backend_buft_get_alignment(reg_ctx->devices[id]->iface.get_buffer_type(reg_ctx->devices[id]));
        col_rounding = std::max(col_rounding, alignment);
    }
    return col_rounding;
}

static void rpc_get_col_split(int64_t * col_low, int64_t * col_high, const ggml_tensor * tensor,
                              const std::array<float, RPC_MAX_DEVICES> & tensor_split, int id) {
    const int64_t ncols = ggml_ncols(tensor);
    rpc_note_split_block(tensor);  // record this weight's quant block in the split grid
    // The column dimension (ne[0]) is the *quantized* dimension, so a split MUST
    // land on a quant-block boundary, else ncols_split is a fractional number of
    // blocks and the per-device byte size is computed inconsistently (ggml_row_size
    // vs ceil-blocks in get_split_col_data vs floor-blocks in nb[1]) -> heap
    // overflow / wrong shape. Buffer byte alignment alone (~32) is finer than a
    // block (256 for Q5_K) so does NOT block-align; max() with the block size does.
    // (N=2 lined up by luck; N>=4 did not.)
    const int64_t rounding = std::max(rpc_get_col_rounding(tensor_split), ggml_blck_size(tensor->type));
    rpc_split_range(ncols, rounding, tensor_split, id, col_low, col_high);
}

static size_t ggml_nbytes_split_col(const struct ggml_tensor * tensor, int ncols_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1] * ggml_row_size(tensor->type, ncols_split);
}

static void ggml_backend_rpc_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_split_buffer_context * ctx = (ggml_backend_rpc_split_buffer_context *) buffer->context;
    delete ctx;
}

static bool ends_with(const char * name, const char * suffix) {
    size_t name_len   = strlen(name);
    size_t suffix_len = strlen(suffix);

    if (suffix_len > name_len) {
        return false;
    }

    return strcmp(name + name_len - suffix_len, suffix) == 0;
}

static void ggml_backend_rpc_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_split_buffer_context *      ctx = (ggml_backend_rpc_split_buffer_context *) buffer->context;
    ggml_backend_rpc_split_buffer_type_context * buft_ctx =
        (ggml_backend_rpc_split_buffer_type_context *) buffer->buft->context;

    // GGML_LOG_INFO("[%s] init split tensor %s, ne0: %ld, ne1:%ld, ne2:%ld, ne3:%ld\n",__func__,tensor->name,tensor->ne[0],tensor->ne[1],tensor->ne[2],tensor->ne[3]);
    if (tensor->extra == NULL) {
        // GGML_LOG_INFO("creating new tensor extra\n");
        ggml_tensor_extra_rpc * extra = new ggml_tensor_extra_rpc();
        ctx->tensor_extras.push_back(extra);

        //TODO:define the split dimension here
        if (ends_with(tensor->name, "attn_output.weight") || ends_with(tensor->name, "ffn_down.weight")) {
            extra->split_dim = 0;
            // GGML_LOG_INFO("split_dim = %d\n",extra->split_dim);
        } else {
            extra->split_dim = 1;
            // GGML_LOG_INFO("split_dim = %d\n",extra->split_dim);
        }

        // Allocate each device's split buffer concurrently -- per-id (disjoint) writes to
        // extra->buffer_ctx[id]/rows[id]; sockets pre-fetched + HELD so the weak_ptr cache
        // can't churn under the workers (see the split upload). RPC_SERIAL_UPLOAD=1 reverts.
        static const bool                      serial_alloc = getenv("RPC_SERIAL_UPLOAD") != nullptr;
        const int                              n_dev        = ggml_backend_rpc_get_device_count();
        std::vector<std::shared_ptr<socket_t>> socks(n_dev);
        for (int id = 0; id < n_dev; ++id) {
            socks[id] = get_socket(((ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context)->endpoint);
        }
        std::vector<std::thread> threads;
        auto                     alloc_split = [&](int id) {
            if (extra->split_dim == 1) {
                int64_t row_low;
                int64_t row_high;
                rpc_get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);
                int64_t nrows_split = row_high - row_low;
                if (nrows_split == 0) {
                    return;
                }
                //it needs first to allocate the buffer on the server (not done at buffer-alloc time)
                size_t                   size    = ggml_nbytes_split(tensor, nrows_split);
                rpc_msg_alloc_buffer_req request = { size };
                rpc_msg_alloc_buffer_rsp response;
                bool status = send_rpc_cmd(socks[id], RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response,
                                                               sizeof(response));
                GGML_ASSERT(status);
                if (response.remote_ptr != 0) {
                    extra->buffer_ctx[id] =
                        new ggml_backend_rpc_buffer_context{ socks[id], nullptr, response.remote_ptr };
                    extra->rows[id] = { row_low, row_high };
                } else {
                    GGML_LOG_INFO("[%s] failed to allocate buffer for tensor %s on device %d\n", __func__, tensor->name,
                                                      id);
                    extra->buffer_ctx[id] = nullptr;
                }
            } else if (extra->split_dim == 0) {
                int64_t col_low;
                int64_t col_high;
                rpc_get_col_split(&col_low, &col_high, tensor, buft_ctx->tensor_split, id);
                int64_t ncols_split = col_high - col_low;
                if (ncols_split == 0) {
                    return;
                }
                size_t                   size    = ggml_nbytes_split_col(tensor, ncols_split);
                rpc_msg_alloc_buffer_req request = { size };
                rpc_msg_alloc_buffer_rsp response;
                bool status = send_rpc_cmd(socks[id], RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response,
                                                               sizeof(response));
                GGML_ASSERT(status);
                if (response.remote_ptr != 0) {
                    extra->buffer_ctx[id] =
                        new ggml_backend_rpc_buffer_context{ socks[id], nullptr, response.remote_ptr };
                    extra->rows[id] = { col_low, col_high };
                } else {
                    GGML_LOG_INFO("[%s] failed to allocate buffer for tensor %s on device %d\n", __func__, tensor->name,
                                                      id);
                    extra->buffer_ctx[id] = nullptr;
                }
            } else {
                GGML_LOG_INFO("[%s] alloc split buffer for non-split tensor %s\n", __func__, tensor->name);
            }
        };
        for (int id = 0; id < n_dev; ++id) {
            if (serial_alloc) {
                alloc_split(id);
            } else {
                threads.emplace_back(alloc_split, id);
            }
        }
        for (auto & t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        tensor->extra = extra;
    }
    for (int id = 0; id < ggml_backend_rpc_get_device_count(); ++id) {
        // CUDA backend on the server pads everything to 512 due to CUDA limitations.
        // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
        // In particular, only quantized tensors need padding
        if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
            // GGML_LOG_INFO("quantized\n");
            rpc_msg_init_tensor_req request;
            request.tensor = split_serialize_tensor(tensor, (ggml_tensor_extra_rpc *) tensor->extra, id);

            ggml_backend_rpc_device_context * dev_ctx =
                (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
            //now use the same command, the server don't know it just has part of the tensor
            bool status =
                send_rpc_cmd(get_socket(dev_ctx->endpoint), RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
            GGML_ASSERT(status);
        }
    }
}

static void * ggml_backend_rpc_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    //Seems not used for split buffer since the base ptr is stored in the context
    return (void *) 0x1000;
    GGML_UNUSED(buffer);
}

static void get_split_col_data(void * output_data, ggml_tensor * tensor, int64_t col_low, int64_t col_high,
                               const void * data) {
    assert(col_low >= 0 && col_high <= tensor->ne[0]);

    // const int64_t ne0 = tensor->ne[0];  // columns
    const int64_t ne1 = tensor->ne[1];  // rows
    const int64_t ne2 = tensor->ne[2];  // depth/batch
    const int64_t ne3 = tensor->ne[3];  // head/group

    int     block_size     = ggml_blck_size(tensor->type);
    int64_t block_col_low  = col_low / block_size;
    int64_t block_col_high = (col_high + block_size - 1) / block_size;
    int64_t out_blocks     = block_col_high - block_col_low;
    // const int64_t out_cols = col_high - col_low;

    const size_t element_size = ggml_type_size(tensor->type);
    const size_t nb0          = tensor->nb[0];  // column stride
    const size_t nb1          = tensor->nb[1];  // row stride
    const size_t nb2          = tensor->nb[2];  // batch stride
    const size_t nb3          = tensor->nb[3];  // 4th-dim stride
    // GGML_LOG_INFO("output_data = %p data = %p\n out_cols = %ld ne1= %ld ne2 = %ld ne3 = %ld",output_data,data,out_cols,ne1,ne2,ne3);

    const uint8_t * input  = (const uint8_t *) data;
    uint8_t *       output = (uint8_t *) output_data;

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                const uint8_t * row_ptr = input + (i3 * nb3) + (i2 * nb2) + (i1 * nb1);
                for (int64_t i0 = block_col_low; i0 < block_col_high; ++i0) {
                    const uint8_t * src = row_ptr + (i0 * nb0);
                    int64_t         out_offset =
                        (((i3 * ne2 + i2) * ne1 + i1) * out_blocks + (i0 - block_col_low)) * element_size;
                    uint8_t * dst = output + out_offset;
                    memcpy(dst, src, element_size);
                }
            }
        }
    }
}

static void ggml_backend_rpc_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                     const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_UNUSED(buffer);
    // GGML_LOG_INFO("[%s] set split tensor %s\n",__func__,tensor->name);
    static bool set_split = false;
    if (split && !set_split) {
        for (int id = 0; id < ggml_backend_rpc_get_device_count(); ++id) {
            auto *                dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
            rpc_msg_set_split_rsp response;
            // GGML_LOG_INFO("[%s] setting split for device %d, endpoint=%s\n", __func__, id, dev_ctx->endpoint.c_str());
            bool                  status =
                send_rpc_cmd(get_socket(dev_ctx->endpoint), RPC_CMD_SET_SPLIT, NULL, 0, &response, sizeof(response));
            GGML_ASSERT(status);
        }
        set_split = true;
    }

    const size_t            nb1   = tensor->nb[1];
    ggml_tensor_extra_rpc * extra = (ggml_tensor_extra_rpc *) tensor->extra;
    std::atomic<size_t>     total_size{ 0 };

    // Upload each device's slice concurrently. Loads are latency-bound (a blocking
    // round-trip per slice), so overlapping them is a win. total_size is atomic and
    // we join before returning, so set_tensor stays synchronous.
    // IMPORTANT: get_socket() returns a shared_ptr from a weak_ptr cache, so a
    // socket only stays alive while a caller holds it. Pre-fetch and HOLD every
    // device's socket here on the main thread for the whole upload -- otherwise the
    // workers race get_socket() and let the cached sockets churn (close/reopen)
    // concurrently, which drops the peer connections and deadlocks the next
    // all-reduce. RPC_SERIAL_UPLOAD=1 forces the old one-at-a-time path.
    static const bool                      serial_upload = getenv("RPC_SERIAL_UPLOAD") != nullptr;
    const int                              n_dev         = ggml_backend_rpc_get_device_count();
    std::vector<std::shared_ptr<socket_t>> socks(n_dev);
    for (int id = 0; id < n_dev; ++id) {
        auto dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
        socks[id]    = get_socket(dev_ctx->endpoint);
    }
    std::vector<std::thread> threads;

    // Weight cache: before uploading a slice, ask the server (by content hash) whether
    // it already has it on disk from a previous run. On a HIT the server loads it from
    // its cache and we skip the network upload; on a MISS we upload via SET_TENSOR_CACHE
    // so the server persists it for next time. RPC_NO_WEIGHT_CACHE disables the cache.
    static const bool weight_cache = (getenv("RPC_NO_WEIGHT_CACHE") == nullptr);

    // Given a contiguous slice, either skip it (server cache hit) or upload it.
    auto cache_or_upload = [&](int id, const rpc_tensor & rt, const uint8_t * slice, size_t slice_size) {
        if (weight_cache) {
            const uint64_t          hash = rpc_fnv1a(slice, slice_size);
            rpc_msg_load_cached_req qreq;
            qreq.tensor = rt;
            qreq.hash   = hash;
            rpc_msg_load_cached_rsp qrsp;
            qrsp.hit     = 0;
            bool qstatus = send_rpc_cmd(socks[id], RPC_CMD_LOAD_CACHED, &qreq, sizeof(qreq), &qrsp, sizeof(qrsp));
            GGML_ASSERT(qstatus);
            if (qrsp.hit) {
                total_size += slice_size;  // server loaded it from its cache -- no upload
                return;
            }
            // miss: | rpc_tensor | offset (8) | hash (8) | data | -> server writes AND caches
            std::vector<uint8_t> input(sizeof(rpc_tensor) + sizeof(offset) + sizeof(hash) + slice_size);
            memcpy(input.data(), &rt, sizeof(rpc_tensor));
            memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
            memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), &hash, sizeof(hash));
            memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset) + sizeof(hash), slice, slice_size);
            bool status = send_rpc_cmd(socks[id], RPC_CMD_SET_TENSOR_CACHE, input.data(), input.size(), nullptr, 0);
            GGML_ASSERT(status);
        } else {
            // | rpc_tensor | offset (8) | data |
            std::vector<uint8_t> input(sizeof(rpc_tensor) + sizeof(offset) + slice_size);
            memcpy(input.data(), &rt, sizeof(rpc_tensor));
            memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
            memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), slice, slice_size);
            bool status = send_rpc_cmd(socks[id], RPC_CMD_SET_TENSOR, input.data(), input.size(), nullptr, 0);
            GGML_ASSERT(status);
        }
        total_size += slice_size;
    };

    auto upload_slice = [&](int id) {
        rpc_tensor rt = split_serialize_tensor(tensor, (ggml_tensor_extra_rpc *) tensor->extra, id);
        if (extra->split_dim == 1) {
            int64_t row_low     = extra->rows[id].first;
            int64_t row_high    = extra->rows[id].second;
            int64_t nrows_split = row_high - row_low;
            if (nrows_split == 0) {
                return;
            }
            const size_t offset_split = row_low * nb1;
            size_t       split_size   = ggml_nbytes_split(tensor, nrows_split);
            cache_or_upload(id, rt, ((const uint8_t *) data) + offset_split, split_size);
        } else if (extra->split_dim == 0) {
            int64_t col_low     = extra->rows[id].first;
            int64_t col_high    = extra->rows[id].second;
            int64_t ncols_split = col_high - col_low;
            if (ncols_split == 0) {
                return;
            }
            size_t               split_size = ggml_nbytes_split_col(tensor, ncols_split);
            std::vector<uint8_t> input_data(split_size, 0);
            get_split_col_data(input_data.data(), tensor, col_low, col_high, data);
            cache_or_upload(id, rt, input_data.data(), split_size);
        } else {
            GGML_LOG_INFO("[%s]set split tensor for non-split tensor %s\n", __func__, tensor->name);
        }
    };

    for (int id = 0; id < n_dev; ++id) {
        if (serial_upload) {
            upload_slice(id);
        } else {
            threads.emplace_back(upload_slice, id);
        }
    }
    for (auto & t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    GGML_ASSERT(total_size == size);
}

static void set_split_col_data(const void * output_data, const ggml_tensor * tensor, int64_t col_low, int64_t col_high,
                               void * data) {
    assert(col_low >= 0 && col_high <= tensor->ne[0]);

    // const int64_t ne0 = tensor->ne[0];  // columns
    const int64_t ne1 = tensor->ne[1];  // rows
    const int64_t ne2 = tensor->ne[2];  // depth/batch
    const int64_t ne3 = tensor->ne[3];  // head/group

    const size_t block_size     = ggml_blck_size(tensor->type);
    int64_t      block_col_low  = col_low / block_size;
    int64_t      block_col_high = (col_high + block_size - 1) / block_size;
    int64_t      out_blocks     = block_col_high - block_col_low;
    // const int64_t out_cols = col_high - col_low;

    const size_t element_size = ggml_type_size(tensor->type);
    const size_t nb0          = tensor->nb[0];  // column stride
    const size_t nb1          = tensor->nb[1];  // row stride
    const size_t nb2          = tensor->nb[2];  // batch stride
    const size_t nb3          = tensor->nb[3];  // 4th-dim stride

    uint8_t *       input  = (uint8_t *) data;
    const uint8_t * output = (const uint8_t *) output_data;

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                uint8_t * row_ptr = input + (i3 * nb3) + (i2 * nb2) + (i1 * nb1);
                for (int64_t i0 = block_col_low; i0 < block_col_high; ++i0) {
                    uint8_t * src = row_ptr + (i0 * nb0);
                    int64_t   out_offset =
                        (((i3 * ne2 + i2) * ne1 + i1) * out_blocks + (i0 - block_col_low)) * element_size;
                    const uint8_t * dst = output + out_offset;
                    memcpy(src, dst, element_size);
                }
            }
        }
    }
}

static void ggml_backend_rpc_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                     void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    // ggml_backend_rpc_split_buffer_type_context * buft_ctx =
    //     (ggml_backend_rpc_split_buffer_type_context *) buffer->buft->context;
    const size_t            nb1   = tensor->nb[1];
    ggml_tensor_extra_rpc * extra = (ggml_tensor_extra_rpc *) tensor->extra;
    std::atomic<size_t>     total_size{ 0 };

    // Download each device's slice concurrently (latency-bound round-trips). Each
    // thread writes a DISJOINT region of `data` (its own row/col range), so the only
    // shared state is the atomic counter. Pre-fetch + HOLD the sockets on the main
    // thread so the weak_ptr cache can't churn under the workers (see the split
    // upload). RPC_SERIAL_UPLOAD=1 reverts to one-at-a-time.
    static const bool                      serial_download = getenv("RPC_SERIAL_UPLOAD") != nullptr;
    const int                              n_dev           = ggml_backend_rpc_get_device_count();
    std::vector<std::shared_ptr<socket_t>> socks(n_dev);
    for (int id = 0; id < n_dev; ++id) {
        socks[id] = get_socket(((ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context)->endpoint);
    }
    std::vector<std::thread> threads;

    auto download_slice = [&](int id) {
        if (extra->split_dim == 1) {
            int64_t row_low     = extra->rows[id].first;
            int64_t row_high    = extra->rows[id].second;
            int64_t nrows_split = row_high - row_low;
            if (nrows_split == 0) {
                return;
            }
            const size_t           offset_split = row_low * nb1;
            size_t                 split_size   = ggml_nbytes_split(tensor, nrows_split);
            rpc_msg_get_tensor_req request;
            request.tensor = split_serialize_tensor(tensor, (ggml_tensor_extra_rpc *) tensor->extra, id);
            request.offset = offset;
            request.size   = split_size;
            bool status    = send_rpc_cmd(socks[id], RPC_CMD_GET_TENSOR, &request, sizeof(request),
                                          (void *) ((uint8_t *) data + offset_split), split_size);
            GGML_ASSERT(status);
            total_size += split_size;
        } else if (extra->split_dim == 0) {
            int64_t col_low     = extra->rows[id].first;
            int64_t col_high    = extra->rows[id].second;
            int64_t ncols_split = col_high - col_low;
            if (ncols_split == 0) {
                return;
            }
            size_t                 split_size = ggml_nbytes_split_col(tensor, ncols_split);
            rpc_msg_get_tensor_req request;
            request.tensor = split_serialize_tensor(tensor, (ggml_tensor_extra_rpc *) tensor->extra, id);
            request.offset = offset;
            request.size   = split_size;
            std::vector<uint8_t> output_data(split_size);
            bool                 status =
                send_rpc_cmd(socks[id], RPC_CMD_GET_TENSOR, &request, sizeof(request), output_data.data(), split_size);
            GGML_ASSERT(status);
            set_split_col_data(output_data.data(), tensor, col_low, col_high, data);  // disjoint cols per device
            total_size += split_size;
        } else {
            GGML_LOG_INFO("[%s] get split tensor for non-split tensor %s\n", __func__, tensor->name);
        }
    };

    for (int id = 0; id < n_dev; ++id) {
        if (serial_download) {
            download_slice(id);
        } else {
            threads.emplace_back(download_slice, id);
        }
    }
    for (auto & t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    GGML_ASSERT(size == total_size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_rpc_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_split_buffer_context * ctx = (ggml_backend_rpc_split_buffer_context *) buffer->context;
    for (ggml_tensor_extra_rpc * extra : ctx->tensor_extras) {
        auto ctx_item = extra->buffer_ctx;
        for (int i = 0; i < RPC_MAX_DEVICES; ++i) {
            if (ctx_item[i]) {
                rpc_msg_buffer_clear_req request = { ctx_item[i]->remote_ptr, value };
                bool                     status =
                    send_rpc_cmd(ctx_item[i]->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
                GGML_ASSERT(status);
            }
        }
    }
}

static const ggml_backend_buffer_i ggml_backend_rpc_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_rpc_split_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *) buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *) buft->context;
    rpc_msg_alloc_buffer_req               request  = { size };
    rpc_msg_alloc_buffer_rsp               response;
    //The question is, we can't store these buffers for the tensor, so hard to load them on every server

    auto sock   = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    GGML_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
            buft, ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{ sock, nullptr, response.remote_ptr }, response.remote_size);
        // GGML_LOG_INFO("[%s] allocated buffer for size %zu, remote_ptr=%" PRIx64 ", remote_size=%" PRIu64 "\n", __func__, size, response.remote_ptr, response.remote_size);
        return buffer;
    }
    return nullptr;
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, nullptr, 0, &response, sizeof(response));
    GGML_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *) buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_get_max_size_rsp response;
    bool                     status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, nullptr, 0, &response, sizeof(response));
    GGML_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *) buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // See comments in init_tensor.
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *) buft->context;
        auto                                   sock     = get_socket(buft_ctx->endpoint);

        rpc_msg_get_alloc_size_req request;

        request.tensor = serialize_tensor(tensor);

        rpc_msg_get_alloc_size_rsp response;
        bool                       status =
            send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        GGML_ASSERT(status);

        return response.alloc_size;
    }
    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

//split buffer type interface
static const char * ggml_backend_rpc_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_split_buffer_type_context * buft_ctx =
        (ggml_backend_rpc_split_buffer_type_context *) buft->context;
    return buft_ctx->name.c_str();
}

static bool ggml_backend_buft_is_rpc(ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    return true;
}

static ggml_backend_buffer_t ggml_backend_rpc_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                             size_t                     size) {
    ggml_backend_rpc_split_buffer_context * ctx = new ggml_backend_rpc_split_buffer_context();
    return ggml_backend_buffer_init(buft, ggml_backend_rpc_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_rpc_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_split_buffer_type_context * buft_ctx =
        (ggml_backend_rpc_split_buffer_type_context *) buft->context;
    return buft_ctx->alignment;
}

static size_t ggml_backend_rpc_split_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_split_buffer_type_context * buft_ctx =
        (ggml_backend_rpc_split_buffer_type_context *) buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                                const ggml_tensor *        tensor) {
    // See comments in init_tensor.
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        size_t total_size = 0;
        for (int id = 0; id < ggml_backend_rpc_get_device_count(); ++id) {
            int64_t row_low;
            int64_t row_high;
            rpc_get_row_split(&row_low, &row_high, tensor,
                              ((ggml_backend_rpc_split_buffer_type_context *) buft->context)->tensor_split, id);
            int64_t nrows_split = row_high - row_low;
            if (nrows_split == 0) {
                continue;
            }
            auto *                     dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
            auto                       sock    = get_socket(dev_ctx->endpoint);
            rpc_msg_get_alloc_size_req request;

            request.tensor = split_serialize_tensor(tensor, (ggml_tensor_extra_rpc *) tensor->extra, id);

            rpc_msg_get_alloc_size_rsp response;
            bool                       status =
                send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
            GGML_ASSERT(status);
            total_size += response.alloc_size;
        }
        return total_size;
    }
    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_split_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_split_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_split_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static ggml_backend_buffer_type_t ggml_backend_rpc_split_buffer_type(int main_device, const float * tensor_split) {
    static std::mutex                                                                                    mutex;
    std::lock_guard<std::mutex>                                                                          lock(mutex);
    static std::map<std::pair<int, std::array<float, RPC_MAX_DEVICES>>, struct ggml_backend_buffer_type> split_buft_map;

    split = true;

    static std::array<float, RPC_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr ||
                    std::all_of(tensor_split, tensor_split + RPC_MAX_DEVICES, [](float x) { return x == 0.0f; });

    if (all_zero) {
        // TODO: tensor_split_arr = ggml_rpc_info().default_tensor_split;
        // For now, we just use equal split
        GGML_LOG_INFO("Computing split\n");
        float split_sum     = 0.0f;
        float default_split = 1.0f / ggml_backend_rpc_get_device_count();
        for (int i = 0; i < ggml_backend_rpc_get_device_count(); ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += default_split;
        }
        for (int i = 0; i < ggml_backend_rpc_get_device_count(); ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    } else {
        GGML_LOG_INFO("Using provided split\n");
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_backend_rpc_get_device_count(); ++i) {
            GGML_LOG_INFO("split_sum: %f, tensor_split: %f\n", split_sum, tensor_split[i]);
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_backend_rpc_get_device_count(); ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    tensor_splits = tensor_split_arr;
    GGML_LOG_INFO("num of devices: %d\n", ggml_backend_rpc_get_device_count());
    GGML_LOG_INFO("tensor splits: ");
    for (int i = 0; i < ggml_backend_rpc_get_device_count(); i++) {
        GGML_LOG_INFO("%f ", tensor_splits[i]);
        GGML_LOG_INFO("%f ", tensor_split_arr[i]);
    }
    GGML_LOG_INFO("\n");
    auto it = split_buft_map.find({ main_device, tensor_split_arr });
    if (it != split_buft_map.end()) {
        return &it->second;
    }
    size_t alignment = 0;
    size_t max_size  = 0;
    for (int i = 0; i < ggml_backend_rpc_get_device_count(); i++) {
        auto * dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[i]->context;
        auto   sock    = get_socket(dev_ctx->endpoint);
        if (sock == nullptr) {
            fprintf(stderr, "Failed to connect to %s\n", dev_ctx->endpoint.c_str());
            return nullptr;
        }
        alignment = std::max(get_alignment(sock), alignment);
        max_size  = std::max(get_max_size(sock), max_size);
    }

    auto * maindev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[main_device]->context;
    ggml_backend_rpc_split_buffer_type_context * buft_ctx = new ggml_backend_rpc_split_buffer_type_context{
        /* .endpoint  = */ maindev_ctx->endpoint,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size,
        /* .tensor_split = */ tensor_splits,
        /* .name      = */ "RPC[" + std::string(maindev_ctx->endpoint) + "]" + "_SPLIT",
    };

    struct ggml_backend_buffer_type buft{ /* .iface   = */ ggml_backend_rpc_split_buffer_type_interface,
                                          /* .device  = */ ggml_backend_rpc_add_device(maindev_ctx->endpoint.c_str()),
                                          /* .context = */ buft_ctx };

    auto result = split_buft_map.emplace(std::make_pair(main_device, tensor_split_arr), buft);
    return &result.first->second;
}

static bool ggml_backend_buft_is_rpc_split(ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_split_buffer_type_name) {
        return false;
    }
    return true;
}

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // this is no-op because we don't have any async operations
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors,
                       std::unordered_set<ggml_tensor *> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    };
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static int change_ne_and_nb(ggml_tensor * tensor, rpc_tensor & rpc_t, std::map<ggml_tensor *, rpc_tensor> & visited,
                            int id) {
    switch (tensor->op) {
        case GGML_OP_NONE:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_CONCAT:
        case GGML_OP_SUM:
            break;
        case GGML_OP_DUP:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SILU_BACK:
        case GGML_OP_NORM:  // normalize
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_DIAG:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_CLAMP:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_LEAKY_RELU:
            {
                //same shape as src0
                rpc_tensor src_tensor = visited[tensor->src[0]];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    rpc_t.ne[i] = src_tensor.ne[i];
                    rpc_t.nb[i] = src_tensor.nb[i];
                }
            }
            break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                //same shape as src0 except ne0
                rpc_tensor src_tensor = visited[tensor->src[0]];
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    rpc_t.ne[i] = src_tensor.ne[i];
                    rpc_t.nb[i] = src_tensor.nb[i];
                }
            }
            break;
        case GGML_OP_REPEAT:
            {
                rpc_tensor src_tensor = visited[tensor->src[0]];
                int        min        = (rpc_t.ne[0] - rpc_t.ne[0] % src_tensor.ne[0]) / src_tensor.ne[0];
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    int curr = (rpc_t.ne[i] - rpc_t.ne[i] % src_tensor.ne[i]) / src_tensor.ne[i];
                    min      = std::min(curr, min);
                }
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    rpc_t.ne[i] = src_tensor.ne[i] * min;
                    if (i == 0) {
                        rpc_t.nb[i] = rpc_t.nb[0];
                    } else if (i == 1) {
                        rpc_t.nb[i] = rpc_t.nb[0] * (rpc_t.ne[0] / ggml_blck_size(tensor->type));
                    } else {
                        rpc_t.nb[i] = rpc_t.nb[i - 1] * rpc_t.ne[i - 1];
                    }
                }
            }
            break;

        case GGML_OP_MUL_MAT:
        case GGML_OP_OUT_PROD:
            {
                rpc_tensor src_tensor0 = visited[tensor->src[0]];
                rpc_tensor src_tensor1 = visited[tensor->src[1]];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    rpc_t.ne[i] = i == 0 ? src_tensor0.ne[1] : src_tensor1.ne[i];
                    if (i == 0) {
                        rpc_t.nb[i] = rpc_t.nb[0];
                    } else if (i == 1) {
                        rpc_t.nb[i] = rpc_t.nb[0] * (rpc_t.ne[0] / ggml_blck_size(tensor->type));
                    } else {
                        rpc_t.nb[i] = rpc_t.nb[i - 1] * rpc_t.ne[i - 1];
                    }
                }
            }
            break;

        case GGML_OP_MUL_MAT_ID:
            {
                rpc_tensor src_tensor0 = visited[tensor->src[0]];
                rpc_tensor src_tensor1 = visited[tensor->src[1]];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (i < 2) {
                        rpc_t.ne[i] = i == 0 ? src_tensor0.ne[1] : src_tensor1.ne[i];
                    }
                    if (i == 0) {
                        rpc_t.nb[i] = rpc_t.nb[0];
                    } else if (i == 1) {
                        rpc_t.nb[i] = rpc_t.nb[0] * (rpc_t.ne[0] / ggml_blck_size(tensor->type));
                    } else {
                        rpc_t.nb[i] = rpc_t.nb[i - 1] * rpc_t.ne[i - 1];
                    }
                }
            }
            break;

        case GGML_OP_CPY:
            {
                rpc_tensor src_tensor = visited[tensor->src[1]];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    rpc_t.ne[i] = src_tensor.ne[i];
                    rpc_t.nb[i] = src_tensor.nb[i];
                }
                //
            }
            break;
        case GGML_OP_CONT:
            {
                rpc_tensor src_tensor = visited[tensor->src[0]];
                rpc_t.ne[0]           = src_tensor.ne[0] * src_tensor.ne[1];
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    rpc_t.nb[i] = i == 1 ? rpc_t.nb[0] * (rpc_t.ne[0] / ggml_blck_size(tensor->type)) :
                                           rpc_t.nb[i - 1] * rpc_t.ne[i - 1];
                }
            }
            break;
        case GGML_OP_RESHAPE:
            {
                rpc_tensor src_tensor = visited[tensor->src[0]];
                rpc_t.ne[1]           = src_tensor.ne[0] / rpc_t.ne[0];
                for (int i = 2; i < GGML_MAX_DIMS; i++) {
                    rpc_t.nb[i] = rpc_t.nb[i - 1] * rpc_t.ne[i - 1];
                }
            }
            break;
        case GGML_OP_VIEW:
            {
                //Assuming that view will always happen after the first split
                if (rpc_t.ne[2] != 1) {
                    int low = rpc_t.ne[2] * tensor_splits[id];
                    int high;
                    if (id == ggml_backend_rpc_get_device_count() - 1) {
                        high = rpc_t.ne[2];
                    } else {
                        high = rpc_t.ne[2] * tensor_splits[id + 1];
                    }
                    rpc_t.ne[2] = high - low;
                    rpc_t.nb[3] = rpc_t.nb[2] * rpc_t.ne[2];
                    if (strncmp(tensor->name, "k", 1) == 0) {
                        rpc_t.nb[1] = rpc_t.nb[3];
                    }
                    if (strncmp(tensor->name, "v", 1) == 0) {
                        float split_part = (id == ggml_backend_rpc_get_device_count() - 1) ?
                                               (1 - tensor_splits[id]) :
                                               (tensor_splits[id + 1] - tensor_splits[id]);
                        rpc_t.nb[1]      = rpc_t.nb[1] * split_part;
                        rpc_t.nb[2]      = rpc_t.nb[1] * rpc_t.ne[1];
                        rpc_t.nb[3]      = rpc_t.nb[2] * rpc_t.ne[2];
                    }

                } else {
                    rpc_tensor & src_tensor = visited[tensor->src[0]];
                    // GGML_LOG_INFO("ne0: %ld ne1: %ld, ne2: %ld ne3: %ld nb0: %ld nb1: %ld, nb2: %ld nb3: %ld",src_tensor.ne[0],src_tensor.ne[1],src_tensor.ne[2],src_tensor.ne[3],src_tensor.nb[0],src_tensor.nb[1],src_tensor.nb[2],src_tensor.nb[3]);
                    float        split_part = (id == ggml_backend_rpc_get_device_count() - 1) ?
                                                  (1 - tensor_splits[id]) :
                                                  (tensor_splits[id + 1] - tensor_splits[id]);
                    // Block-align the activation's contraction slice to the weight's
                    // quant grid (g_rpc_split_block) so it matches the column-split
                    // weight it feeds; raw split_part*ne[0] only lined up at N=2.
                    // Guard: only when each device gets >= one block (small/per-head
                    // views keep the fine-grained float split).
                    {
                        const int64_t devs = ggml_backend_rpc_get_device_count();
                        const int64_t full = src_tensor.ne[0];
                        if (devs > 0 && full / devs >= g_rpc_split_block) {
                            const int64_t align = std::max(rpc_get_col_rounding(tensor_splits), g_rpc_split_block);
                            src_tensor.ne[0]    = rpc_split_count(full, align, tensor_splits, id);
                        } else {
                            src_tensor.ne[0] = split_part * full;
                        }
                    }
                    src_tensor.nb[1] = src_tensor.ne[0] * src_tensor.nb[0] / ggml_blck_size(tensor->src[0]->type);
                    src_tensor.nb[2] = src_tensor.ne[1] * src_tensor.nb[1];
                    src_tensor.nb[3] = src_tensor.ne[2] * src_tensor.nb[2];
                    if (rpc_t.ne[1] != 1) {
                        int64_t row_low;
                        int64_t row_high;
                        rpc_get_row_split(&row_low, &row_high, tensor, tensor_splits, id);
                        // GGML_LOG_INFO("low: %ld high: %ld\n",row_low,row_high);
                        rpc_t.ne[1] = row_high - row_low;
                        rpc_t.nb[2] = rpc_t.nb[1] * rpc_t.ne[1];
                        rpc_t.nb[3] = rpc_t.nb[2] * rpc_t.ne[2];
                    } else if (rpc_t.ne[0] != 1) {
                        int64_t col_low;
                        int64_t col_high;
                        rpc_get_col_split(&col_low, &col_high, tensor, tensor_splits, id);
                        // GGML_LOG_INFO("low: %ld high: %ld\n",col_low,col_high);
                        rpc_t.ne[0] = col_high - col_low;
                        for (int i = 1; i < GGML_MAX_DIMS; i++) {
                            rpc_t.nb[i] = i == 1 ? rpc_t.nb[0] * (rpc_t.ne[0] / ggml_blck_size(tensor->type)) :
                                                   rpc_t.nb[i - 1] * rpc_t.ne[i - 1];
                        }

                    } else {
                        GGML_LOG_INFO("error view");
                    }
                    if (strncmp(tensor->name, "v", 1) == 0) {
                        float split_part = (id == ggml_backend_rpc_get_device_count() - 1) ?
                                               (1 - tensor_splits[id]) :
                                               (tensor_splits[id + 1] - tensor_splits[id]);
                        rpc_t.nb[1]      = rpc_t.nb[1] * split_part;
                        rpc_t.nb[2]      = rpc_t.nb[1] * rpc_t.ne[1];
                        rpc_t.nb[3]      = rpc_t.nb[2] * rpc_t.ne[2];
                    }
                    return 0;
                }
            }
            break;

        case GGML_OP_PERMUTE:
            {
                ggml_tensor * src        = tensor->src[0];
                rpc_tensor    src_tensor = visited[tensor->src[0]];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    for (int j = 0; j < GGML_MAX_DIMS; j++) {
                        if (tensor->ne[i] == src->ne[j]) {
                            rpc_t.ne[i] = src_tensor.ne[j];
                            rpc_t.nb[i] = src_tensor.nb[j];
                            continue;
                        }
                    }
                }
            }
            break;
        case GGML_OP_TRANSPOSE:
            {
                rpc_tensor src_tensor = visited[tensor->src[0]];
                rpc_t.ne[1]           = src_tensor.ne[0];
                rpc_t.ne[0]           = src_tensor.ne[1];
                rpc_t.nb[1]           = src_tensor.nb[0];
                rpc_t.nb[0]           = src_tensor.nb[1];
                for (int i = 2; i < GGML_MAX_DIMS; i++) {
                    rpc_t.nb[i] = src_tensor.nb[i];
                }
            }
            break;
        case GGML_OP_GET_ROWS:
            {
                rpc_tensor src_tensor0 = visited[tensor->src[0]];
                rpc_tensor src_tensor1 = visited[tensor->src[1]];
                rpc_t.ne[0]            = src_tensor0.ne[0];
                rpc_t.ne[1]            = src_tensor1.ne[0];
                rpc_t.ne[2]            = src_tensor1.ne[1];
                rpc_t.ne[3]            = src_tensor1.ne[2];
            }
            break;
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
        case GGML_OP_UPSCALE:  // nearest interpolate
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:

        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_COUNT:
            break;

        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_GET_REL_POS:
        case GGML_OP_ADD_REL_POS:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:

        case GGML_OP_UNARY:

        case GGML_OP_MAP_UNARY:
        case GGML_OP_MAP_BINARY:

        case GGML_OP_MAP_CUSTOM1_F32:
        case GGML_OP_MAP_CUSTOM2_F32:
        case GGML_OP_MAP_CUSTOM3_F32:

        case GGML_OP_MAP_CUSTOM1:
        case GGML_OP_MAP_CUSTOM2:
        case GGML_OP_MAP_CUSTOM3:

        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
            {
                //same shape as src0
                rpc_tensor src_tensor = visited[tensor->src[0]];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    rpc_t.ne[i] = src_tensor.ne[i];
                    rpc_t.nb[i] = src_tensor.nb[i];
                }
            }
            break;
    }

    return -1;
}

static void add_tensor_part(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors,
                            std::map<ggml_tensor *, rpc_tensor> & visited, int split_dim, int id) {
    if (tensor == nullptr || visited.count(tensor)) {
        return;
    }
    int src0_idx = -1;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        ggml_tensor * src = tensor->src[i];
        rpc_tensor    src_tensor;
        rpc_tensor    src_view_tensor;
        if (src && visited.count(src) == 0) {
            if (split_dim != -1 && i == 0) {
                src_tensor = split_serialize_tensor(src, (ggml_tensor_extra_rpc *) src->extra, id);
                // tensors.push_back(src_tensor);

                if (src->view_src && visited.count(src->view_src) == 0) {
                    src_view_tensor =
                        split_serialize_tensor(src->view_src, (ggml_tensor_extra_rpc *) src->view_src->extra, id);
                    tensors.push_back(src_view_tensor);
                    visited[src->view_src] = src_view_tensor;
                }
            } else {
                src_tensor                        = serialize_tensor(src);
                ggml_tensor_extra_rpc * src_extra = (ggml_tensor_extra_rpc *) src->extra;
                if (src_extra->buffer_ctx[id] == nullptr) {
                    GGML_LOG_INFO("[%s] buffer context for device %d is null\n", __func__, id);
                } else {
                    if (src_extra->buffer_ctx[id]->remote_ptr != src_tensor.buffer) {
                        src_tensor.buffer = src_extra->buffer_ctx[id]->remote_ptr;
                        src_tensor.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                            reinterpret_cast<ggml_backend_rpc_buffer_context *>(src_extra->buffer_ctx[id])));
                        if (src->op == GGML_OP_VIEW || src->op == GGML_OP_CPY) {
                            uint64_t offset = src->view_offs;
                            if (strncmp(src->name, "k", 1) == 0) {
                                float split_part = (id == ggml_backend_rpc_get_device_count() - 1) ?
                                                       (1 - tensor_splits[id]) :
                                                       (tensor_splits[id + 1] - tensor_splits[id]);
                                offset           = src->view_offs * split_part;
                            }
                            src_tensor.data += offset;
                            src_tensor.view_offs = offset;
                            // GGML_LOG_INFO("data after view_offs: %ld\n",src_tensor.data);
                        }
                    } else {
                        if ((src->op == GGML_OP_VIEW || src->op == GGML_OP_CPY) && strncmp(src->name, "k", 1) == 0) {
                            float    split_part = (id == ggml_backend_rpc_get_device_count() - 1) ?
                                                      (1 - tensor_splits[id]) :
                                                      (tensor_splits[id + 1] - tensor_splits[id]);
                            uint64_t offset     = src->view_offs * (1 - split_part);
                            src_tensor.data -= offset;
                            src_tensor.view_offs = src->view_offs * split_part;
                        }
                    }
                }
                // tensors.push_back(src_tensor);

                if (src->view_src && visited.count(src->view_src) == 0) {
                    src_view_tensor                            = serialize_tensor(src->view_src);
                    ggml_tensor_extra_rpc * src_view_src_extra = (ggml_tensor_extra_rpc *) src->view_src->extra;
                    if (src_view_src_extra->buffer_ctx[id] == nullptr) {
                        GGML_LOG_INFO("[%s] buffer context for device %d is null\n", __func__, id);
                    } else {
                        if (src_view_tensor.buffer != src_view_src_extra->buffer_ctx[id]->remote_ptr) {
                            src_view_tensor.buffer = src_view_src_extra->buffer_ctx[id]->remote_ptr;
                            src_view_tensor.data = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                                reinterpret_cast<ggml_backend_rpc_buffer_context *>(
                                    src_view_src_extra->buffer_ctx[id])));
                        }
                    }
                    tensors.push_back(src_view_tensor);
                    visited[src->view_src] = src_view_tensor;
                }
            }
            visited[src] = src_tensor;
            if (i == 0) {
                src0_idx = tensors.size();
            }
            tensors.push_back(src_tensor);
            // tensors.push_back(src_view_tensor);
        }
    }

    rpc_tensor              rpc_t        = serialize_tensor(tensor);
    ggml_tensor_extra_rpc * tensor_extra = (ggml_tensor_extra_rpc *) tensor->extra;
    if (tensor_extra->buffer_ctx[id] == nullptr) {
        GGML_LOG_INFO("[%s] buffer context for device %d is null\n", __func__, id);
    } else {
        if (rpc_t.buffer != tensor_extra->buffer_ctx[id]->remote_ptr) {
            rpc_t.buffer = tensor_extra->buffer_ctx[id]->remote_ptr;
            rpc_t.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                reinterpret_cast<ggml_backend_rpc_buffer_context *>(tensor_extra->buffer_ctx[id])));
            if (tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_CPY) {
                uint64_t offset = tensor->view_offs;
                if (strncmp(tensor->name, "k", 1) == 0) {
                    float split_part = (id == ggml_backend_rpc_get_device_count() - 1) ?
                                           (1 - tensor_splits[id]) :
                                           (tensor_splits[id + 1] - tensor_splits[id]);
                    offset           = tensor->view_offs * split_part;
                }
                rpc_t.data += offset;
                rpc_t.view_offs = offset;
                // GGML_LOG_INFO("data after view_offs: %ld\n",rpc_t.data);
            }
        } else {
            if ((tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_CPY) && strncmp(tensor->name, "k", 1) == 0) {
                float    split_part = (id == ggml_backend_rpc_get_device_count() - 1) ?
                                          (1 - tensor_splits[id]) :
                                          (tensor_splits[id + 1] - tensor_splits[id]);
                uint64_t offset     = tensor->view_offs * (1 - split_part);
                rpc_t.data -= offset;
                rpc_t.view_offs = tensor->view_offs * split_part;
            }
        }
    }

    int checksrc = change_ne_and_nb(tensor, rpc_t, visited, id);
    if (checksrc != -1) {
        tensors[src0_idx] = visited[tensor->src[0]];
    }
    tensors.push_back(rpc_t);
    visited[tensor] = rpc_t;

    struct rpc_tensor view_tensor;
    if (tensor->view_src && visited.count(tensor->view_src) == 0) {
        view_tensor                            = serialize_tensor(tensor->view_src);
        ggml_tensor_extra_rpc * view_src_extra = (ggml_tensor_extra_rpc *) tensor->view_src->extra;
        if (view_src_extra->buffer_ctx[id] == nullptr) {
            GGML_LOG_INFO("[%s] buffer context for device %d is null\n", __func__, id);
        } else {
            if (view_tensor.buffer != view_src_extra->buffer_ctx[id]->remote_ptr) {
                view_tensor.buffer = view_src_extra->buffer_ctx[id]->remote_ptr;
                view_tensor.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                    reinterpret_cast<ggml_backend_rpc_buffer_context *>(view_src_extra->buffer_ctx[id])));
            }
        }
        visited[tensor->view_src] = view_tensor;
        tensors.push_back(view_tensor);
    }
}

static void serialize_graph(const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor>           tensors;
    std::unordered_set<ggml_tensor *> visited;
    // GGML_LOG_INFO("begin serialize graph, n_nodes = %d\n",n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        // ggml_tensor * node = cgraph->nodes[i];
        // GGML_LOG_INFO("\ni = %d\n", i);
        // GGML_LOG_INFO("\ntensor %s ne0 :%ld ne1: %ld ne2: %ld ne3: %ld nb0: %ld nb1: %ld nb2: %ld nb3: %ld ",
        //             node->name,node->ne[0],node->ne[1],node->ne[2],node->ne[3],node->nb[0],node->nb[1],node->nb[2],node->nb[3]);
        // GGML_LOG_INFO("Operation: %d\n",node->op);
        // for(int i=0;i<GGML_MAX_SRC;i++){
        //     ggml_tensor* src=node->src[i];
        //     if(src){
        //         GGML_LOG_INFO("src %d %s ne0 :%ld ne1: %ld ne2: %ld ne3: %ld nb0: %ld nb1: %ld nb2: %ld nb3: %ld \n",
        //                         i,src->name,src->ne[0],src->ne[1],src->ne[2],src->ne[3],src->nb[0],src->nb[1],src->nb[2],src->nb[3]);
        //     }
        // }
        add_tensor(cgraph->nodes[i], tensors, visited);
        // GGML_LOG_INFO("\nadd\n");
    }
    // GGML_LOG_INFO("finish add tensor in graph\n");
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors   = tensors.size();
    int      output_size = sizeof(uint32_t) + (n_nodes * sizeof(uint64_t)) + sizeof(uint32_t) +
                      (n_tensors * sizeof(rpc_tensor));  //+sizeof(bool);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, sizeof(n_nodes));
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + sizeof(n_nodes) + (i * sizeof(uint64_t)), &cgraph->nodes[i], sizeof(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.data() + sizeof(n_nodes) + (n_nodes * sizeof(uint64_t)));
    *out_ntensors           = n_tensors;
    rpc_tensor * out_tensors =
        (rpc_tensor *) (output.data() + sizeof(n_nodes) + (n_nodes * sizeof(uint64_t)) + sizeof(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static void ggml_compute_forward_add_f32(struct ggml_tensor * dst, void * dst_data, void * src0_data,
                                         void * src1_data) {
    const struct ggml_tensor * src0 = dst;
    const struct ggml_tensor * src1 = dst;

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    const int ith = 0;
    const int nth = 1;

    const int nr = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = (ir - (i03 * ne02 * ne01) - (i02 * ne01));

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst_data + (i03 * nb3) + (i02 * nb2) + (i01 * nb1));
            float * src0_ptr = (float *) ((char *) src0_data + (i03 * nb03) + (i02 * nb02) + (i01 * nb01));
            float * src1_ptr = (float *) ((char *) src1_data + (i13 * nb13) + (i12 * nb12) + (i11 * nb11));

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                vDSP_vadd(src0_ptr + r * ne10, 1, src1_ptr, 1, dst_ptr + r * ne10, 1, ne10);
#else
                ggml_vec_add_f32(ne10, dst_ptr + r * ne10, src0_ptr + r * ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = (ir - (i03 * ne02 * ne01) - (i02 * ne01));

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst_data + (i03 * nb3) + (i02 * nb2) + (i01 * nb1));
            float * src0_ptr = (float *) ((char *) src0_data + (i03 * nb03) + (i02 * nb02) + (i01 * nb01));

            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int64_t i10 = i0 % ne10;
                float *       src1_ptr =
                    (float *) ((char *) src1_data + (i13 * nb13) + (i12 * nb12) + (i11 * nb11) + (i10 * nb10));

                dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
            }
        }
    }
}

static void add_data_to_data(std::vector<uint8_t> & data, ggml_tensor * tensor, std::mutex & data_mutex, int id) {
    std::lock_guard<std::mutex> lock(data_mutex);
    ggml_tensor_extra_rpc *     src_extra = (ggml_tensor_extra_rpc *) tensor->src[0]->extra;
    ggml_tensor_extra_rpc *     extra     = (ggml_tensor_extra_rpc *) tensor->extra;
    if (src_extra->split_dim == 1) {
        int64_t col_low  = src_extra->rows[id].first;
        int64_t col_high = src_extra->rows[id].second;

        int64_t ncols_split = col_high - col_low;
        if (ncols_split == 0) {
            return;
        }
        size_t                 split_size = ggml_nbytes_split_col(tensor, ncols_split);
        rpc_msg_get_tensor_req request;
        request.tensor = serialize_tensor(tensor);
        if (request.tensor.buffer != extra->buffer_ctx[id]->remote_ptr) {
            request.tensor.buffer = extra->buffer_ctx[id]->remote_ptr;
            request.tensor.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                reinterpret_cast<ggml_backend_rpc_buffer_context *>(extra->buffer_ctx[id])));
        }
        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            if (i == 0) {
                request.tensor.ne[i] = col_high - col_low;  //ne1 is the number of elements in a column, that is, rows
            } else {
                request.tensor.ne[i] = tensor->ne[i];
            }
            if (i == 0) {
                request.tensor.nb[i] = tensor->nb[i];  //recalculate nb after ne changes
            } else if (i == 1) {
                request.tensor.nb[1] = request.tensor.nb[0] * (request.tensor.ne[0] / ggml_blck_size(tensor->type));
            } else {
                request.tensor.nb[i] = request.tensor.nb[i - 1] * request.tensor.ne[i - 1];
            }
        }
        request.offset                            = 0;
        request.size                              = split_size;
        ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
        std::vector<uint8_t>              output_data(split_size);
        bool status = send_rpc_cmd(get_socket(dev_ctx->endpoint), RPC_CMD_GET_TENSOR, &request, sizeof(request),
                                   output_data.data(), split_size);
        GGML_ASSERT(status);
        set_split_col_data(output_data.data(), tensor, col_low, col_high, data.data());
    } else {
        auto                  dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
        ggml_backend_buffer_t buf     = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

        GGML_ASSERT(data.size() == ggml_nbytes(tensor));
        size_t                  buf_size = ggml_tensor_overhead() * (1 + 3) + ggml_graph_overhead_custom(3, false);
        struct ggml_init_params params   = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/true,
        };
        struct ggml_context * ctx   = ggml_init(params);
        struct ggml_cgraph *  graph = ggml_new_graph_custom(ctx, 3, false);
        graph->n_nodes              = 0;
        ggml_tensor * temp =
            ggml_new_tensor_4d(ctx, tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        strncpy(temp->name, "temp", 5);
        buf                                 = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        ggml_backend_buffer_type_t buft     = buf->buft;
        bool                       same_dev = buft == ggml_backend_rpc_buffer_type(dev_ctx->endpoint.c_str());
        if (!same_dev) {
            buft = ggml_backend_rpc_buffer_type(dev_ctx->endpoint.c_str());
        }

        temp->buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(tensor));
        temp->data   = buf->iface.get_base(temp->buffer);
        multi_cpy    = false;
        buf->iface.init_tensor(temp->buffer, temp);
        buf->iface.set_tensor(temp->buffer, temp, data.data(), 0, data.size());

        ggml_tensor * tensor_cpy =
            ggml_new_tensor_4d(ctx, tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        if (!same_dev) {
            tensor_cpy->buffer = ggml_backend_buffer_init(buft, ggml_backend_rpc_buffer_interface,
                                                          extra->buffer_ctx[id], tensor->buffer->size);
            tensor_cpy->data   = ggml_backend_rpc_buffer_context_get_base(extra->buffer_ctx[id]);
        } else {
            tensor_cpy->buffer = tensor->buffer;
            tensor_cpy->data   = tensor->data;
        }
        strncpy(tensor_cpy->name, "tensor_cpy", 11);

        ggml_tensor * add_out = ggml_add(ctx, tensor_cpy, temp);
        add_out->buffer       = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(tensor));
        add_out->data         = buf->iface.get_base(add_out->buffer);
        buf->iface.init_tensor(add_out->buffer, add_out);

        multi_cpy = true;
        ggml_build_forward_expand(graph, add_out);

        std::vector<uint8_t> input;
        serialize_graph(graph, input);
        rpc_msg_graph_compute_rsp response;
        auto                      sock = get_socket(dev_ctx->endpoint);
        bool                      status =
            send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size(), &response, sizeof(response));
        GGML_ASSERT(status);
        buf->iface.get_tensor(add_out->buffer, add_out, data.data(), 0, data.size());

        ggml_backend_buffer_free(temp->buffer);
        ggml_backend_buffer_free(add_out->buffer);
        ggml_free(ctx);
    }
}

/**
 * @brief Computes a computation graph on an RPC backend, optionally splitting the graph for parallel execution.
 *
 * This function executes the given computation graph (`cgraph`) using a remote procedure call (RPC) backend.
 * If the `split` flag is enabled, the graph is partitioned into subgraphs based on tensor split dimensions,
 * and each subgraph is computed concurrently across multiple devices using threads. The results from each device
 * are then aggregated and written back to the appropriate tensor buffers.
 *
 * If `split` is not enabled, the entire graph is serialized and sent as a single RPC command for remote execution.
 *
 * @param backend  The backend context to use for computation (must be an RPC backend).
 * @param cgraph   The computation graph to execute.
 * @return         GGML_STATUS_SUCCESS on success, or an error status on failure.
 *
 * @note
 * - The function assumes that the backend context and devices are properly initialized.
 * - When splitting is enabled, the function identifies split points in the graph based on tensor operations
 *   and split dimensions, and synchronizes results as needed.
 * - Thread safety is ensured via mutexes when aggregating results from multiple devices.
 * - The function asserts on RPC failures and unexpected tensor states.
 */
static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // GGML_LOG_INFO("graph compute for cgraph %x\n", (uint64_t) cgraph);
    static std::unordered_map<uint64_t, uint8_t> graph_splits;  //{graph: number}
    static uint8_t                               global_graph_number = 0;
    ggml_backend_rpc_context *                   rpc_ctx             = (ggml_backend_rpc_context *) backend->context;

    struct sync_split {
        std::pair<uint32_t, uint32_t> nodes_split;
        bool                          checkend;
    };
    static struct sync_split change_split;
    // GGML_LOG_INFO("[%s] computing graph with %d nodes\n", __func__, cgraph->n_nodes);
    if (split) {
        //check wether the graph is stored at the server
        // if(graph_splits.find((uint64_t)cgraph)==graph_splits.end())
        {
            // Find next synchronization point
            std::vector<struct sync_split> sync_splits;
            uint32_t                       count_nodes_low = 0;
            for (int count_nodes = 0; count_nodes < cgraph->n_nodes; count_nodes++) {
                ggml_tensor * node = cgraph->nodes[count_nodes];
                // if(node->op==GGML_OP_VIEW){
                //     GGML_LOG_INFO("\ntensor %s ne0 :%ld ne1: %ld ne2: %ld ne3: %ld nb0: %ld nb1: %ld nb2: %ld nb3: %ld \n",
                //         node->name,node->ne[0],node->ne[1],node->ne[2],node->ne[3],node->nb[0],node->nb[1],node->nb[2],node->nb[3]);
                //     GGML_LOG_INFO("view_offs %ld\n",node->view_offs);
                // }
                // GGML_LOG_INFO("\ntensor %s ne0 :%ld ne1: %ld ne2: %ld ne3: %ld nb0: %ld nb1: %ld nb2: %ld nb3: %ld \n",
                //         node->name,node->ne[0],node->ne[1],node->ne[2],node->ne[3],node->nb[0],node->nb[1],node->nb[2],node->nb[3]);
                // for(int i=0;i<GGML_MAX_SRC;i++){
                //     ggml_tensor* src=node->src[i];
                //     if(src){
                //         GGML_LOG_INFO("src %d %s ne0 :%ld ne1: %ld ne2: %ld ne3: %ld nb0: %ld nb1: %ld nb2: %ld nb3: %ld \n",
                //                 i,src->name,src->ne[0],src->ne[1],src->ne[2],src->ne[3],src->nb[0],src->nb[1],src->nb[2],src->nb[3]);
                //     }
                // }
                if (!ggml_is_empty(node) && node->src[0] != nullptr &&
                    ggml_backend_buft_is_rpc_split(node->src[0]->buffer->buft) &&
                    (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) {
                    ggml_tensor_extra_rpc * node_extra = (ggml_tensor_extra_rpc *) node->src[0]->extra;
                    if (node_extra->split_dim == 0) {
                        // this is where to break;
                        GGML_ASSERT(ggml_backend_buft_is_rpc(node->src[1]->buffer->buft));
                        GGML_ASSERT(ggml_backend_buft_is_rpc(node->buffer->buft));
                        struct sync_split sync_split = {
                            { count_nodes_low, count_nodes },
                            false
                        };
                        sync_splits.push_back(sync_split);
                        count_nodes_low = count_nodes + 1;
                        // GGML_LOG_INFO("\n-----another split-----\n");
                    } else if (node_extra->split_dim == 1) {
                        GGML_ASSERT(ggml_backend_buft_is_rpc(node->src[1]->buffer->buft));
                        GGML_ASSERT(ggml_backend_buft_is_rpc(node->buffer->buft));
                        if (strcmp(node->name, "result_output") == 0) {
                            struct sync_split sync_split = {
                                { count_nodes_low, count_nodes },
                                true
                            };
                            sync_splits.push_back(sync_split);
                        }
                    }
                } else {
                    if (count_nodes == cgraph->n_nodes - 1) {
                        struct sync_split sync_split = {
                            { count_nodes_low, count_nodes },
                            true
                        };
                        sync_splits.push_back(sync_split);
                        // GGML_LOG_INFO("\n-----end of subgraph-----\n");
                    }
                }
            }
            change_split = sync_splits[sync_splits.size() - 2];
            // Compute the computation graph for the current split and synchronize results
            for (size_t count_split = 0; count_split < sync_splits.size(); count_split++) {
                // GGML_LOG_INFO("\n------------split-------------\n");
                struct sync_split sync_split      = sync_splits[count_split];
                uint32_t          count_nodes_low = sync_split.nodes_split.first;
                uint32_t          count_nodes     = sync_split.nodes_split.second;
                // GGML_LOG_INFO("low = %d high = %d\n",count_nodes_low,count_nodes);
                // bool              checkend        = sync_split.checkend;

                ggml_tensor * tensor = cgraph->nodes[count_nodes];

                //compute concurrently
                std::mutex rpc_mutex;
                std::mutex data_mutex;

                int                      device_count = ggml_backend_rpc_get_device_count();
                std::vector<std::thread> threads;

                for (int id = 0; id < device_count; ++id) {
                    threads.emplace_back([&, id]() {
                        //we need to compute the next part of the graph
                        std::vector<rpc_tensor>             tensors;
                        std::map<ggml_tensor *, rpc_tensor> visited;

                        for (uint32_t count = count_nodes_low; count <= count_nodes; count++) {
                            ggml_tensor * node = cgraph->nodes[count];
                            if (!ggml_is_empty(node) && node->src[0] != nullptr &&
                                ggml_backend_buft_is_rpc_split(node->src[0]->buffer->buft) &&
                                (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) {
                                // This node is eligible for RPC splitting
                                ggml_tensor_extra_rpc * node_extra = (ggml_tensor_extra_rpc *) node->src[0]->extra;
                                add_tensor_part(node, tensors, visited, node_extra->split_dim, id);
                            } else {
                                // This node is not eligible for RPC splitting
                                add_tensor_part(node, tensors, visited, -1, id);
                            }
                        }

                        auto dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
                        auto sock    = get_socket(dev_ctx->endpoint);

                        std::vector<uint8_t> input;
                        uint32_t             n_nodes = count_nodes - count_nodes_low + 1;
                        if (n_nodes == 0) {
                            return;
                        }

                        //input format: signal(1 byte) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t)) |
                        //              n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) | graph_number (1 byte)
                        uint32_t n_tensors  = tensors.size();
                        int      input_size = sizeof(uint8_t) + sizeof(uint32_t) + n_nodes * sizeof(uint64_t) +
                                         sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor) + sizeof(uint8_t);
                        input.resize(input_size, 0);

                        //add a signal value to indicate the type of all-reduce at the beginning of the input
                        //0: addition
                        //1: concatenation (only for result_output, so maybe just at the client)
                        //2: no operation (nothing to do at the client and servers)
                        ggml_tensor_extra_rpc * tensor_extra = (ggml_tensor_extra_rpc *) tensor->src[0]->extra;
                        uint8_t                 signal = tensor_extra->split_dim == -1 ? 2 : tensor_extra->split_dim;
                        memcpy(input.data(), &signal, sizeof(uint8_t));

                        // Serialize input graph for RPC command
                        // Append number of nodes
                        memcpy(input.data() + sizeof(uint8_t), &n_nodes, sizeof(n_nodes));
                        for (uint32_t i = 0; i < n_nodes; i++) {
                            // Copy each node pointer (as uint64_t) into the input buffer for serialization
                            memcpy(input.data() + sizeof(uint8_t) + sizeof(n_nodes) + i * sizeof(uint64_t),
                                   &cgraph->nodes[count_nodes_low + i], sizeof(uint64_t));
                        }

                        // Append number of tensors
                        uint32_t * in_ntensors = (uint32_t *) (input.data() + sizeof(uint8_t) + sizeof(n_nodes) +
                                                               n_nodes * sizeof(uint64_t));
                        *in_ntensors           = n_tensors;

                        // Copy tensor metadata
                        rpc_tensor * in_tensors = (rpc_tensor *) (input.data() + sizeof(uint8_t) + sizeof(n_nodes) +
                                                                  n_nodes * sizeof(uint64_t) + sizeof(uint32_t));
                        memcpy(in_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));

                        // denote which graph number this is, and info for graph that servers need to know
                        uint8_t * in_graph_number =
                            (uint8_t *) (input.data() + sizeof(uint8_t) + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) +
                                         sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor));
                        *in_graph_number = global_graph_number;

                        rpc_msg_graph_compute_rsp response;

                        //here the graph_compute commend actually not do the compute, but store the graph
                        //we will have another commend that will tell the servers to compute
                        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size(), &response,
                                                   sizeof(response));
                        GGML_ASSERT(status);

                        if (response.result != GGML_STATUS_SUCCESS) {
                            std::lock_guard<std::mutex> lock(rpc_mutex);  // To avoid mixed output
                            fprintf(stderr, "RPC graph compute failed with status %d\n", response.result);
                            return;
                        }
                    });
                }

                // Join all threads
                for (auto & thread : threads) {
                    if (thread.joinable()) {
                        thread.join();
                    }
                }
            }

            graph_splits[reinterpret_cast<uint64_t>(cgraph)] = global_graph_number;
            global_graph_number++;
        }
        // else{
        //     GGML_LOG_INFO("graph %ld has been computed before with graph number %d, skip computation\n",
        //                   (uint64_t)cgraph, graph_splits[reinterpret_cast<uint64_t>(cgraph)]);
        // }

        //send a commend to servers to ask them do the computation job here
        int                      device_count = ggml_backend_rpc_get_device_count();
        std::vector<std::thread> threads;

        ggml_tensor *        tensor = cgraph->nodes[cgraph->n_nodes - 1];
        std::vector<uint8_t> data(ggml_nbytes(tensor), 0);
        std::mutex           data_mutex;
        for (int id = 0; id < device_count; ++id) {
            threads.emplace_back([&, id]() {
                auto dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
                auto sock    = get_socket(dev_ctx->endpoint);

                rpc_msg_do_computation_req compute_info;
                compute_info.graph_number = graph_splits[reinterpret_cast<uint64_t>(cgraph)];
                bool status =
                    send_rpc_cmd(sock, RPC_CMD_DO_COMPUTATION, &compute_info, sizeof(compute_info), nullptr, 0);
                GGML_ASSERT(status);

                if (strcmp(tensor->name, "result_output") == 0) {
                    // GGML_LOG_INFO("getting result data from device %d\n", id);
                    add_data_to_data(data, tensor, data_mutex, id);
                }
            });
        }
        // Join all threads
        for (auto & thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        // Write back data to RPC backend tensor on the client

        if (strcmp(tensor->name, "result_output") == 0) {
            ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
            buf->iface.set_tensor(buf, tensor, data.data(), 0, data.size());
        }
        return GGML_STATUS_SUCCESS;
    } else {
        // Non-split (single device / no -sm row): always compute the graph
        // inline on the server and return its status -- the stock llama.cpp RPC
        // behavior. The graph_splits cache + DO_COMPUTATION path is only valid
        // for the tensor-parallel store-then-execute flow above (the server
        // stores graphs there); in non-split mode the server never stores, so a
        // cached DO_COMPUTATION would have nothing to run.
        std::vector<uint8_t> input;
        serialize_graph(cgraph, input);
        rpc_msg_graph_compute_rsp response;
        auto                      sock = get_socket(rpc_ctx->endpoint);
        bool                      status =
            send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size(), &response, sizeof(response));
        GGML_ASSERT(status);
        return (enum ggml_status) response.result;
    }
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint) {
    static std::mutex                                                  mutex;
    std::lock_guard<std::mutex>                                        lock(mutex);
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto                                                               it = buft_map.find(endpoint);
    if (it != buft_map.end()) {
        // GGML_LOG_INFO("FIND\n");
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        fprintf(stderr, "Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t                                 alignment = get_alignment(sock);
    size_t                                 max_size  = get_max_size(sock);
    ggml_backend_rpc_buffer_type_context * buft_ctx =
        new ggml_backend_rpc_buffer_type_context{ /* .endpoint  = */ endpoint,
                                                  /* .name      = */ "RPC[" + std::string(endpoint) + "]",
                                                  /* .alignment = */ alignment,
                                                  /* .max_size  = */ max_size };

    ggml_backend_buffer_type_t buft =
        new ggml_backend_buffer_type{ /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
                                      /* .device  = */ ggml_backend_rpc_add_device(endpoint),
                                      /* .context = */ buft_ctx };
    buft_map[endpoint] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint) {
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context{
        /* .endpoint  = */ endpoint,
        /* .name      = */ "RPC[" + std::string(endpoint) + "]",
    };

    ggml_backend_t backend = new ggml_backend{ /* .guid      = */ ggml_backend_rpc_guid(),
                                               /* .interface = */ ggml_backend_rpc_interface,
                                               /* .device    = */ ggml_backend_rpc_add_device(endpoint),
                                               /* .context   = */ ctx };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, nullptr, 0, &response, sizeof(response));
    GGML_ASSERT(status);
    *free  = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free  = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, free, total);
}

// RPC server-side implementation

// All-reduce block for aggregating tensors from multiple clients
class all_reduce_block {
  public:
    all_reduce_block() {}

    all_reduce_block(ggml_tensor * tensor, int op, int device_count, ggml_backend_t backend);
    ~all_reduce_block();
    bool block_init(ggml_tensor * tensor, int op, int device_count, ggml_backend_t backend, uint8_t device_id);
    bool add(std::vector<uint8_t> & input, uint8_t device_id);

    bool add_to_buffer(std::vector<uint8_t> input) {
        std::lock_guard<std::mutex> lock(add_mutex);
        all_reduce_buffer.push_back(input);
        return true;
    }

    //reset the block for the next-time use
    void block_uinit() {
        initialized  = false;
        arrived      = 0;
        is_completed = false;
        ggml_backend_buffer_free(add_tensor->buffer);
        ggml_free(ctx);
    }

    bool is_init() { return initialized; }

    void wait_for_completion();

    std::vector<std::vector<uint8_t>> & get_all_reduce_buffer() { return all_reduce_buffer; }
  private:
    bool                              initialized = false;    //whether the tensor to be reduced has been set
    ggml_cgraph *                     graph       = nullptr;  //addition graph
    ggml_tensor *                     tensor;                 //tensor to be reduced
    ggml_tensor *                     add_tensor;             //data collected from other servers
    ggml_tensor *                     tensor_out;
    int                               op;                     //reduction operation
    int                               num_of_servers;         //number of servers to wait for
    std::mutex                        add_mutex;              //mutex for adding data
    std::mutex                        wait_mutex;             //mutex for waiting completion
    std::condition_variable           cv;                     //condition variable for signaling
    bool                              is_completed = false;   //whether the all-reduce is completed
    int                               arrived      = 0;       //number of servers that have received data
    std::vector<std::vector<uint8_t>> all_reduce_buffer;      //buffer for storing data added before initialization
    ggml_backend_t                    backend;                //backend type
    struct ggml_context *             ctx;
};

void all_reduce_block::wait_for_completion() {
    std::unique_lock<std::mutex> lock(wait_mutex);
    cv.wait(lock, [this] { return is_completed; });
}

//function description: create the all_reduce_block for a specific tensor and initialize it by the current server, create the add tensor and graph
//when to be called:    when the current server finishes graph computing
//note:                 since the first time to be called, no data in the buffer
all_reduce_block::all_reduce_block(ggml_tensor * tensor, int op, int device_count, ggml_backend_t backend) :
    op(op),
    num_of_servers(device_count),
    backend(backend) {
    // GGML_LOG_INFO("all_reduce_block called\n");
    //set tensor to be reduced
    this->tensor = tensor;

    //create addition graph
    size_t                  buf_size = ggml_tensor_overhead() * (1 + 3) + ggml_graph_overhead_custom(3, false);
    struct ggml_init_params params   = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ctx                        = ggml_init(params);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 3, false);
    graph->n_nodes             = 0;
    add_tensor = ggml_new_tensor_4d(ctx, tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    strncpy(add_tensor->name, "add_tensor", 11);
    ggml_backend_buffer_type_t buft   = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t      buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(tensor));
    add_tensor->buffer                = buffer;
    if (buffer == nullptr) {
        GGML_LOG_INFO("empty buffer\n");
    }
    add_tensor->data = buffer->iface.get_base(buffer);
    // GGML_LOG_INFO("init tensor\n");
    //no init_tensor function for cpu backend
    // buffer->iface.init_tensor(buffer, add_tensor);

    ggml_tensor * tensor_out = ggml_add(ctx, tensor, add_tensor);
    tensor_out->buffer       = tensor->buffer;
    tensor_out->data         = tensor->data;
    strncpy(tensor_out->name, "tensor_out", 11);

    //if using expand, it will visit all parents, exceeding graph capacity
    // ggml_build_forward_expand(graph, tensor_out);
    ggml_graph_add_node(graph, tensor_out);

    this->graph = graph;

    //set initialized flag
    initialized = true;

    //increment arrived count
    arrived++;

    //add any buffered data (no need as this is constructor, the first time to be called)
    // if(!all_reduce_buffer.empty()){
    //     for(auto & b : all_reduce_buffer){
    //         add(b);
    //
    //     }
    // }
}

//function description: initialize the block with updating tensor
//when to be called:    when the server receiving all_reduce msg from other servers
//note: only update pointer to tensor and create buffer for add if graph exists, else create the graph
bool all_reduce_block::block_init(ggml_tensor * tensor, int op, int device_count, ggml_backend_t backend,
                                  uint8_t device_id) {
    // GGML_LOG_INFO("all_reduce_block init called\n");
    if (initialized) {
        return true;
    }

    //initialize the block
    // GGML_LOG_INFO("initializing the block\n");
    this->op             = op;
    this->num_of_servers = device_count;
    this->backend        = backend;
    this->tensor         = tensor;

    //create addition graph
    size_t                  buf_size = ggml_tensor_overhead() * (1 + 3) + ggml_graph_overhead_custom(3, false);
    struct ggml_init_params params   = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ctx                        = ggml_init(params);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 3, false);
    graph->n_nodes             = 0;
    add_tensor = ggml_new_tensor_4d(ctx, tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    strncpy(add_tensor->name, "add_tensor", 11);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    add_tensor->buffer              = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(tensor));
    add_tensor->data                = add_tensor->buffer->iface.get_base(add_tensor->buffer);
    // add_tensor->buffer->iface.init_tensor(add_tensor->buffer, add_tensor);

    tensor_out         = ggml_add(ctx, tensor, add_tensor);
    tensor_out->buffer = tensor->buffer;
    tensor_out->data   = tensor->data;
    strncpy(tensor_out->name, "tensor_out", 11);

    ggml_graph_add_node(graph, tensor_out);

    this->graph = graph;

    //set initialized flag
    initialized = true;

    //increment arrived count
    arrived++;

    //add data in the buffer if not empty
    if (!all_reduce_buffer.empty()) {
        for (auto it = all_reduce_buffer.begin(); it != all_reduce_buffer.end(); /* no increment here */) {
            add(*it, device_id);
            it = all_reduce_buffer.erase(it);
        }
    }
    return true;
}

all_reduce_block::~all_reduce_block() {
    if (!initialized) {
        // block_uinit() already freed add_tensor->buffer and ctx (and add_tensor
        // itself lives inside that freed ctx) -- freeing again is a use-after-free.
        return;
    }
    ggml_backend_buffer_free(add_tensor->buffer);
    //TODO: does the graph and context need to be freed?
    if (ctx != nullptr) {
        ggml_free(ctx);
    }
}

bool all_reduce_block::add(std::vector<uint8_t> & input, uint8_t device_id) {
    // GGML_LOG_INFO("all_reduce_block add called\n");
    std::lock_guard<std::mutex> lock(add_mutex);

    std::vector<uint8_t> origin;
    origin.resize(input.size(), 0);
    ggml_backend_tensor_get(tensor, origin.data(), 0, origin.size());

    //do the addition
    ggml_backend_tensor_set(add_tensor, input.data(), 0, input.size());
    ggml_status result = ggml_backend_graph_compute(backend, graph);
    if (result != GGML_STATUS_SUCCESS) {
        GGML_LOG_INFO("graph_compute failed\n");
    }

    std::vector<uint8_t> output;
    output.resize(input.size(), 0);
    ggml_backend_tensor_get(tensor, output.data(), 0, output.size());

    arrived++;

    //if all_reduce is done, notify the main thread that waiting for all_reduce result
    if (arrived >= num_of_servers) {
        is_completed = true;
        initialized  = false;
        cv.notify_all();
    }
    GGML_UNUSED(device_id);
    return is_completed;
}

struct graph_info {
    uint8_t        graph_number;
    ggml_cgraph *  cgraph;
    ggml_context * ctx;
    uint8_t        signal;
    graph_info *   next = nullptr;
};

class graph_compute_info {
  public:
    graph_compute_info(uint8_t graph_number, ggml_cgraph * cgraph, ggml_context * ctx, uint8_t signal);
    ~graph_compute_info();
    bool         add_info(ggml_cgraph * cgraph, ggml_context * ctx, uint8_t signal);
    bool         replace_info(ggml_cgraph * cgraph, ggml_context * ctx, uint8_t signal);
    graph_info * head = nullptr;
    graph_info * tail = nullptr;
    uint8_t      graph_number;
    uint8_t      graph_count = 0;
};

graph_compute_info::graph_compute_info(uint8_t graph_number, ggml_cgraph * cgraph, ggml_context * ctx, uint8_t signal) {
    this->graph_number = graph_number;
    graph_info * info  = new graph_info();
    info->graph_number = graph_number;
    info->cgraph       = cgraph;
    info->ctx          = ctx;
    info->signal       = signal;
    head               = info;
    tail               = info;
    graph_count        = 1;
}

bool graph_compute_info::add_info(ggml_cgraph * cgraph, ggml_context * ctx, uint8_t signal) {
    graph_info * info  = new graph_info();
    info->graph_number = graph_number;
    info->cgraph       = cgraph;
    info->ctx          = ctx;
    info->signal       = signal;
    tail->next         = info;
    tail               = info;
    graph_count++;
    return true;
}

bool graph_compute_info::replace_info(ggml_cgraph * cgraph, ggml_context * ctx, uint8_t signal) {
    ggml_free(tail->ctx);
    tail->graph_number = graph_number;
    tail->cgraph       = cgraph;
    tail->ctx          = ctx;
    tail->signal       = signal;
    return true;
}

graph_compute_info::~graph_compute_info() {
    graph_info * current = head;
    while (current != nullptr) {
        graph_info * next = current->next;
        ggml_free(current->ctx);
        delete current;
        current = next;
    }
}

class rpc_server {
  public:
    rpc_server(ggml_backend_t backend) : backend(backend) {}

    ~rpc_server();

    void alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    void get_alignment(rpc_msg_get_alignment_rsp & response);
    void get_max_size(rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool set_split(rpc_msg_set_split_rsp & response);
    bool create_peer_connection(const rpc_msg_create_peer_connection_req & request,
                                rpc_msg_create_peer_connection_rsp &       response);
    void add_socket_listen(const std::shared_ptr<socket_t> & sock);
    bool all_reduce(std::vector<uint8_t> & input);
    bool do_computation(const rpc_msg_do_computation_req & request);
    bool load_cached(const rpc_msg_load_cached_req & request, rpc_msg_load_cached_rsp & response);
    bool set_tensor_cache(const std::vector<uint8_t> & input);

    ggml_backend_t & get_backend() { return backend; }
  private:
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id, struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor *> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor *> &     tensor_map);
    void store_graph_compute_info(uint8_t graph_number, ggml_cgraph * cgraph, ggml_context * ctx, uint8_t signal);
    ggml_backend_t                                           backend;
    std::unordered_set<ggml_backend_buffer_t>                buffers;
    bool                                                     server_split = false;
    std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets_connectto;  //sockets that the server connects to
    // Hold a shared_ptr to each dialed peer socket so the weak_ptrs above don't
    // expire. Otherwise create_peer_connection's local shared_ptr is released
    // immediately and EVERY all-reduce re-dials its peers (a TCP handshake per
    // peer, per layer, per token) -- the dominant decode cost over WiFi.
    std::vector<std::shared_ptr<socket_t>>                   peer_socks_held;
    std::vector<std::weak_ptr<socket_t>>                     sockets_listento;   //sockets that the server listen to
    std::mutex                                               sockets_mutex;      //mutex for adding sockets to the list
    uint8_t                                                  device_id;          //device id for current server
    uint8_t                                                  device_count;       //total num of servers
    std::unordered_map<std::string, all_reduce_block *>      all_reduce_blocks;  //blocks for all reduce
    std::mutex                                               block_mutex;        //mutex for adding or checking blocks
    std::unordered_map<uint8_t, graph_compute_info *> graph_compute_infos;  // map graph_number to graph_compute_info
};

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params    params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };

    struct ggml_context * ctx    = ggml_init(params);
    ggml_tensor *         tensor = deserialize_tensor(ctx, &request.tensor);

    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        ggml_free(ctx);
        return false;
    }

    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backend);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    ggml_free(ctx);
    return true;
}

void rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    ggml_backend_buffer_type_t buft   = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t      buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr               = 0;
    response.remote_size              = 0;
    if (buffer != nullptr) {
        response.remote_ptr  = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        buffers.insert(buffer);
    } else {
        GGML_LOG_INFO("[%s] size: %" PRIu64 " -> failed\n", __func__, request.size);
    }
}

void rpc_server::get_alignment(rpc_msg_get_alignment_rsp & response) {
    ggml_backend_buffer_type_t buft      = ggml_backend_get_default_buffer_type(backend);
    size_t                     alignment = ggml_backend_buft_get_alignment(buft);
    GGML_PRINT_DEBUG("[%s] alignment: %lu\n", __func__, alignment);
    response.alignment = alignment;
}

void rpc_server::get_max_size(rpc_msg_get_max_size_rsp & response) {
    ggml_backend_buffer_type_t buft     = ggml_backend_get_default_buffer_type(backend);
    size_t                     max_size = ggml_backend_buft_get_max_size(buft);
    GGML_PRINT_DEBUG("[%s] max_size: %lu\n", __func__, max_size);
    response.max_size = max_size;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base       = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    ggml_tensor * result =
        ggml_new_tensor_4d(ctx, (ggml_type) tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    // GGML_LOG_INFO("[%s] tensor name: %s, type: %d, ne: [%d, %d, %d, %d], buffer: %p, data: %p\n",
    //     __func__, tensor->name, tensor->type,
    //     tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
    //     (void*)result->buffer, (void*)tensor->data);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        GGML_LOG_INFO("[%s] buffer not found: %p\n", __func__, (void *) result->buffer);
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size  = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size  = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        // GGML_LOG_INFO("[%s] tensor name: %s, buffer: %p, data: %" PRIx64 ", size: %" PRIu64 ", buffer_start: %" PRIx64
        //               ", buffer_size: %" PRIu64 "\n",
        //               __func__, tensor->name, (void *) result->buffer, tensor->data, tensor_size, buffer_start,
        //               buffer_size);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data);  // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start);
        GGML_ASSERT(tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data  = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}

// ---- on-disk weight cache (server side) -----------------------------------
// Persist each uploaded weight slice to disk so reruns skip the network upload.
// Keyed by a content hash the client computes. Dir: $RPC_WEIGHT_CACHE_DIR, else
// $HOME/.cache/llama-rpc-weights. Disable entirely with RPC_NO_WEIGHT_CACHE.
static bool rpc_weight_cache_enabled() {
    static const bool on = (getenv("RPC_NO_WEIGHT_CACHE") == nullptr);
    return on;
}

static const std::string & rpc_weight_cache_dir() {
    static const std::string dir = [] {
        const char * env  = getenv("RPC_WEIGHT_CACHE_DIR");
        const char * home = getenv("HOME");
        std::string  d    = env ? std::string(env) : std::string(home ? home : ".") + "/.cache/llama-rpc-weights";
        // mkdir -p (POSIX, best-effort -- EEXIST is fine)
        std::string  cur;
        for (size_t i = 0; i < d.size(); ++i) {
            cur += d[i];
            if ((d[i] == '/' && cur.size() > 1) || i + 1 == d.size()) {
                mkdir(cur.c_str(), 0755);
            }
        }
        return d;
    }();
    return dir;
}

static std::string rpc_weight_cache_path(uint64_t hash) {
    char name[20];
    snprintf(name, sizeof(name), "%016llx", (unsigned long long) hash);
    return rpc_weight_cache_dir() + "/" + name;
}

// LOAD_CACHED: if the slice with this content hash is on disk, load it straight
// into the destination buffer and report hit=1; otherwise hit=0 (client uploads).
bool rpc_server::load_cached(const rpc_msg_load_cached_req & request, rpc_msg_load_cached_rsp & response) {
    response.hit = 0;
    if (!rpc_weight_cache_enabled()) {
        return true;
    }
    std::ifstream f(rpc_weight_cache_path(request.hash), std::ios::binary | std::ios::ate);
    if (!f) {
        return true;  // miss
    }
    const std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> data((size_t) std::max<std::streamsize>(n, 0));
    if (n <= 0 || !f.read((char *) data.data(), n)) {
        return true;  // miss (unreadable / empty)
    }
    struct ggml_init_params params{ ggml_tensor_overhead(), NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *           tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || (size_t) n != ggml_nbytes(tensor)) {
        // size/shape mismatch -> treat as a miss so the client re-uploads (cache stays safe)
        ggml_free(ctx);
        return true;
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, (size_t) n);
    ggml_free(ctx);
    static const bool dbg_wcache = (getenv("RPC_DBG_WCACHE") != nullptr);
    if (dbg_wcache) {
        GGML_LOG_INFO("[wcache] HIT   %016llx (%lld bytes)\n", (unsigned long long) request.hash, (long long) n);
    }
    response.hit = 1;
    return true;
}

// SET_TENSOR_CACHE: | rpc_tensor | offset(8) | hash(8) | data |. Writes the buffer
// exactly like SET_TENSOR, then persists the slice to the cache file keyed by hash.
bool rpc_server::set_tensor_cache(const std::vector<uint8_t> & input) {
    if (input.size() < sizeof(rpc_tensor) + (2 * sizeof(uint64_t))) {
        GGML_LOG_INFO("[%s] input size too small: %zu\n", __func__, input.size());
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *) input.data();
    uint64_t           offset;
    uint64_t           hash;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    memcpy(&hash, input.data() + sizeof(rpc_tensor) + sizeof(offset), sizeof(hash));
    const size_t size = input.size() - sizeof(rpc_tensor) - (2 * sizeof(uint64_t));
    const void * data = input.data() + sizeof(rpc_tensor) + (2 * sizeof(uint64_t));

    struct ggml_init_params params{ ggml_tensor_overhead(), NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *           tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        GGML_LOG_INFO("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    // sanitize tensor->data (same bounds check as set_tensor)
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_ABORT("[%s] tensor->data out of bounds\n", __func__);
        }
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    ggml_free(ctx);

    // persist to the on-disk cache (best-effort; a failed write just misses next run)
    if (rpc_weight_cache_enabled()) {
        std::ofstream out(rpc_weight_cache_path(hash), std::ios::binary | std::ios::trunc);
        if (out) {
            out.write((const char *) data, size);
        }
        static const bool dbg_wcache = (getenv("RPC_DBG_WCACHE") != nullptr);
        if (dbg_wcache) {
            GGML_LOG_INFO("[wcache] STORE %016llx (%zu bytes)\n", (unsigned long long) hash, size);
        }
    }
    return true;
}

bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        GGML_LOG_INFO("[%s] input size too small: %zu\n", __func__, input.size());
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *) input.data();
    uint64_t           offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx    = ggml_init(params);
    ggml_tensor *         tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        GGML_LOG_INFO("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void *) tensor->buffer,
                     tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_ABORT("[%s] tensor->data out of bounds\n", __func__);
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    ggml_backend_tensor_set(tensor, data, offset, size);
    ggml_free(ctx);
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };

    struct ggml_context * ctx    = ggml_init(params);
    ggml_tensor *         tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        ggml_free(ctx);
        return false;
    }

    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        GGML_LOG_ERROR("Null buffer for tensor passed to init_tensor function\n");
    }

    //now we use tensor->extra for split_buffer
    // if (tensor->extra != nullptr) {
    //     // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
    //     // Currently unimplemented.
    //     GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
    //     ggml_free(ctx);
    //     return false;
    // }

    ggml_free(ctx);
    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx    = ggml_init(params);
    ggml_tensor *         tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__,
                     (void *) tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 || request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
            GGML_ABORT("[%s] tensor->data out of bounds\n", __func__);
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    ggml_free(ctx);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params{
        /*.mem_size   =*/2 * ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor *         src = deserialize_tensor(ctx, &request.src);
    ggml_tensor *         dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        ggml_free(ctx);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_PRINT_DEBUG(
            "[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
            "    write range : [0x%" PRIx64 ", 0x%" PRIx64
            "]\n"
            "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
            __func__, dst_data, dst_data + src_size, dst_base, dst_base + dst_buf_sz);
        ggml_free(ctx);
        return false;
    }

    GGML_PRINT_DEBUG("[%s] src->buffer: %p, dst->buffer: %p\n", __func__, (void *) src->buffer, (void *) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    ggml_free(ctx);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id, struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor *> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor *> &     tensor_map) {
    try {
        if (id == 0) {
            return nullptr;
        }
        if (tensor_map.find(id) != tensor_map.end()) {
            return tensor_map[id];
        }
        const rpc_tensor * tensor = tensor_ptrs.at(id);
        // GGML_LOG_INFO(
        //     "create node with tensor: %s, ne0: %d, ne1: %d, ne2: %d, ne3: %d, nb0: %d, nb1: %d, nb2: %d, nb3: %d\n",
        //     tensor->name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[0], tensor->nb[1],
        //     tensor->nb[2], tensor->nb[3]);

        struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
        // GGML_LOG_INFO("create node with tensor: %s\n", result->name);
        if (result == nullptr) {
            return nullptr;
        }
        tensor_map[id] = result;
        if (server_split) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                uint64_t src_id = tensor->src[i];
                if (src_id == 0) {
                    result->src[i] = nullptr;
                    continue;
                }
                if (tensor_map.find(src_id) != tensor_map.end()) {
                    result->src[i] = tensor_map[src_id];
                    continue;
                }
                const rpc_tensor *   src_tensor = tensor_ptrs.at(src_id);
                struct ggml_tensor * src_result = deserialize_tensor(ctx, src_tensor);
                if (src_result == nullptr) {
                    result->src[i] = nullptr;
                    continue;
                }
                tensor_map[src_id] = src_result;
                result->src[i]     = src_result;

                src_id = src_tensor->view_src;
                if (src_id == 0) {
                    src_result->view_src  = nullptr;
                    src_result->view_offs = src_tensor->view_offs;
                    continue;
                }
                if (tensor_map.find(src_id) != tensor_map.end()) {
                    src_result->view_src  = tensor_map[src_id];
                    src_result->view_offs = src_tensor->view_offs;
                    continue;
                }

                const rpc_tensor *   src_view_tensor = tensor_ptrs.at(src_id);
                struct ggml_tensor * src_view_result = deserialize_tensor(ctx, src_view_tensor);
                if (src_view_result == nullptr) {
                    src_result->view_src  = nullptr;
                    src_result->view_offs = src_tensor->view_offs;
                    continue;
                }
                tensor_map[src_id]    = src_view_result;
                src_result->view_src  = src_view_result;
                src_result->view_offs = src_tensor->view_offs;
            }
            uint64_t src_id = tensor->view_src;
            if (src_id == 0) {
                result->view_src  = nullptr;
                result->view_offs = tensor->view_offs;
                return result;
            }
            if (tensor_map.find(src_id) != tensor_map.end()) {
                result->view_src  = tensor_map[src_id];
                result->view_offs = tensor->view_offs;
                return result;
            }

            const rpc_tensor *   src_tensor = tensor_ptrs.at(src_id);
            struct ggml_tensor * src_result = deserialize_tensor(ctx, src_tensor);
            if (src_result == nullptr) {
                result->view_src  = nullptr;
                result->view_offs = tensor->view_offs;
                return result;
            }
            tensor_map[src_id] = src_result;
            result->view_src   = src_result;
            result->view_offs  = tensor->view_offs;
        } else {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            }
            result->view_src  = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
            result->view_offs = tensor->view_offs;
        }

        return result;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("[%s] tensor %lu with not found in tensor_ptrs: %s\n", __func__, id, e.what());
        return nullptr;
    }
}

void rpc_server::store_graph_compute_info(uint8_t graph_number, ggml_cgraph * cgraph, ggml_context * ctx,
                                          uint8_t signal) {
    auto it = graph_compute_infos.find(graph_number);
    if (it == graph_compute_infos.end()) {
        graph_compute_info * info         = new graph_compute_info(graph_number, cgraph, ctx, signal);
        graph_compute_infos[graph_number] = info;
    } else {
        it->second->add_info(cgraph, ctx, signal);
    }
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response) {
    // Non-split / single-device path: the client only sends RPC_CMD_SET_SPLIT
    // (which sets server_split) in tensor-parallel (-sm row) mode. Without it,
    // the client uses the stock serialize_graph layout (no leading signal byte,
    // no trailing graph_number) and expects the graph COMPUTED inline with its
    // status returned -- standard llama.cpp RPC. The split path below instead
    // STORES the graph for a later DO_COMPUTATION + peer all-reduce.
    if (!server_split) {
        // stock format: | n_nodes (4) | nodes (n_nodes*8) | n_tensors (4) | tensors (n_tensors*rpc_tensor) |
        if (input.size() < sizeof(uint32_t)) {
            return false;
        }
        uint32_t n_nodes;
        memcpy(&n_nodes, input.data(), sizeof(n_nodes));
        if (input.size() < sizeof(uint32_t) + (n_nodes * sizeof(uint64_t)) + sizeof(uint32_t)) {
            return false;
        }
        const uint64_t * nodes = (const uint64_t *) (input.data() + sizeof(n_nodes));
        uint32_t         n_tensors;
        memcpy(&n_tensors, input.data() + sizeof(n_nodes) + (n_nodes * sizeof(uint64_t)), sizeof(n_tensors));
        if (input.size() <
            sizeof(uint32_t) + (n_nodes * sizeof(uint64_t)) + sizeof(uint32_t) + (n_tensors * sizeof(rpc_tensor))) {
            return false;
        }
        const rpc_tensor * tensors =
            (const rpc_tensor *) (input.data() + sizeof(n_nodes) + (n_nodes * sizeof(uint64_t)) + sizeof(n_tensors));
        size_t buf_size = (ggml_tensor_overhead() * (n_nodes + n_tensors)) + ggml_graph_overhead_custom(n_nodes, false);
        struct ggml_init_params params = { /*.mem_size=*/buf_size, /*.mem_buffer=*/NULL, /*.no_alloc=*/true };
        struct ggml_context *   ctx    = ggml_init(params);
        struct ggml_cgraph *    graph  = ggml_new_graph_custom(ctx, n_nodes, false);
        graph->n_nodes                 = n_nodes;
        std::unordered_map<uint64_t, const rpc_tensor *> tensor_ptrs;
        for (uint32_t i = 0; i < n_tensors; i++) {
            tensor_ptrs[tensors[i].id] = &tensors[i];
        }
        std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
        try {
            for (uint32_t i = 0; i < n_nodes; i++) {
                int64_t id;
                memcpy(&id, &nodes[i], sizeof(id));
                graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
            }
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("[%s] exception during node creation: %s\n", __func__, e.what());
            ggml_free(ctx);
            return false;
        }
        ggml_status status = ggml_backend_graph_compute(backend, graph);
        response.result    = status;
        ggml_free(ctx);
        return true;
    }

    // serialization format:
    // signal (1 byte) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) | graph_number (1 byte)
    // GGML_LOG_INFO("graph compute called with input size: %zu\n", input.size());
    if (input.size() < sizeof(uint32_t)) {
        return false;
    }

    //first interprete signal that whether needs all reduce
    uint8_t signal;
    memcpy(&signal, input.data(), sizeof(uint8_t));

    uint32_t n_nodes;
    memcpy(&n_nodes, input.data() + sizeof(uint8_t), sizeof(n_nodes));
    if (input.size() < sizeof(uint8_t) + sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *) (input.data() + sizeof(uint8_t) + sizeof(n_nodes));
    uint32_t         n_tensors;
    memcpy(&n_tensors, input.data() + sizeof(uint8_t) + sizeof(n_nodes) + n_nodes * sizeof(uint64_t),
           sizeof(n_tensors));
    if (input.size() < sizeof(uint8_t) + sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) +
                           n_tensors * sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *) (input.data() + sizeof(uint8_t) + sizeof(n_nodes) +
                                                       n_nodes * sizeof(uint64_t) + sizeof(n_tensors));

    uint8_t graph_number;
    memcpy(&graph_number,
           input.data() + sizeof(uint8_t) + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) + sizeof(n_tensors) +
               n_tensors * sizeof(rpc_tensor),
           sizeof(graph_number));
    size_t buf_size = ggml_tensor_overhead() * (n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx   = ggml_init(params);
    struct ggml_cgraph *  graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes              = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor *> tensor_ptrs;
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    // Print all entries in tensor_ptrs
    // for (const auto & pair : tensor_ptrs) {
    //     const rpc_tensor * t = pair.second;
    // }

    std::unordered_map<uint64_t, ggml_tensor *> tensor_map;
    try {
        for (uint32_t i = 0; i < n_nodes; i++) {
            int64_t id;
            memcpy(&id, &nodes[i], sizeof(id));
            graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("[%s] exception during node creation: %s\n", __func__, e.what());
        ggml_free(ctx);
        return false;
    }

    //store graph compute info
    store_graph_compute_info(graph_number, graph, ctx, signal);
    GGML_LOG_INFO("stored graph compute info for graph number %d\n", graph_number);
    response.result = GGML_STATUS_SUCCESS;
    return true;
}

void compare_node(ggml_tensor * node, ggml_tensor * node_to_compare) {
    std::string   filename = "compare.txt";
    std::ofstream outfile;
    outfile.open(filename, std::ios_base::app);  // append instead of overwrite
    if (node->buffer != node_to_compare->buffer) {
        outfile << "node " << node->name << " and node " << node_to_compare->name << " are different\n";
        outfile << "buffer different: " << node->buffer << " vs " << node_to_compare->buffer << "\n";
    }
    if (node->ne[0] != node_to_compare->ne[0] || node->ne[1] != node_to_compare->ne[1] ||
        node->ne[2] != node_to_compare->ne[2] || node->ne[3] != node_to_compare->ne[3]) {
        outfile << "node " << node->name << " and node " << node_to_compare->name << " are different\n";
        outfile << "ne different: [" << node->ne[0] << "," << node->ne[1] << "," << node->ne[2] << "," << node->ne[3]
                << "] vs [" << node_to_compare->ne[0] << "," << node_to_compare->ne[1] << "," << node_to_compare->ne[2]
                << "," << node_to_compare->ne[3] << "]\n";
    }
    if (node->nb[0] != node_to_compare->nb[0] || node->nb[1] != node_to_compare->nb[1] ||
        node->nb[2] != node_to_compare->nb[2] || node->nb[3] != node_to_compare->nb[3]) {
        outfile << "node " << node->name << " and node " << node_to_compare->name << " are different\n";
        outfile << "nb different: [" << node->nb[0] << "," << node->nb[1] << "," << node->nb[2] << "," << node->nb[3]
                << "] vs [" << node_to_compare->nb[0] << "," << node_to_compare->nb[1] << "," << node_to_compare->nb[2]
                << "," << node_to_compare->nb[3] << "]\n";
    }

    if (node->view_src != node_to_compare->view_src || node->view_offs != node_to_compare->view_offs) {
        outfile << "node " << node->name << " and node " << node_to_compare->name << " are different\n";
        outfile << "view_src or view_offs different: " << node->view_src << " vs " << node_to_compare->view_src << ", "
                << node->view_offs << " vs " << node_to_compare->view_offs << "\n";
        outfile << "view_src difference: " << (uint64_t) node->view_src - (uint64_t) node_to_compare->view_src
                << ", view_off difference: " << node->view_offs - node_to_compare->view_offs << "\n";
    }
    outfile.close();
}

void compare_two_graph(ggml_cgraph * graph, ggml_cgraph * graph_to_compare) {
    for (int i = 0; i < graph->n_nodes; i++) {
        compare_node(graph->nodes[i], graph_to_compare->nodes[i]);
    }
}

bool rpc_server::do_computation(const rpc_msg_do_computation_req & request) {
    // GGML_LOG_INFO("do computation called\n");
    uint8_t           graph_number   = request.graph_number;
    // Debug-only A/B graph comparison: walks every node of graph N vs N-2 and
    // appends diffs to compare.txt -- pure overhead on every even graph in the
    // hot path (and the file grows unbounded, ~13 MB/run). Off by default; set
    // RPC_DBG_COMPARE=1 to re-enable.
    static const bool compare_graphs = getenv("RPC_DBG_COMPARE") != nullptr;
    if (compare_graphs && graph_number % 2 == 0 && graph_number >= 2) {
        GGML_LOG_INFO("graph number is %d, doing comparason\n", graph_number);
        std::string   filename = "compare.txt";
        std::ofstream outfile;
        outfile.open(filename, std::ios_base::app);
        outfile << "Comparason for graph number " << std::to_string(graph_number) << "\n";
        outfile.close();
        auto it = graph_compute_infos.find(graph_number);
        if (it == graph_compute_infos.end()) {
            GGML_LOG_INFO("graph number %d not found\n", graph_number);
            return false;
        }
        auto compare = graph_compute_infos.find(graph_number - 2);
        if (compare == graph_compute_infos.end()) {
            GGML_LOG_INFO("graph number %d not found\n", graph_number);
            return false;
        }
        graph_info * info         = it->second->head;
        graph_info * compare_info = compare->second->head;
        while (info) {
            ggml_cgraph * graph          = info->cgraph;
            ggml_cgraph * compared_graph = compare_info->cgraph;
            compare_two_graph(graph, compared_graph);
            info         = info->next;
            compare_info = compare_info->next;
        }
        GGML_LOG_INFO("comparason done for graph number %d\n", graph_number);
    }
    auto it = graph_compute_infos.find(graph_number);
    if (it == graph_compute_infos.end()) {
        GGML_LOG_INFO("graph number %d not found\n", graph_number);
        return false;
    }
    GGML_LOG_INFO("found graph number %d, doing computation\n", graph_number);
    graph_info * info = it->second->head;
    while (info) {
        ggml_cgraph *  graph  = info->cgraph;
        ggml_context * ctx    = info->ctx;
        uint8_t        signal = info->signal;

        ggml_tensor * tensor_to_all_reduce = graph->nodes[graph->n_nodes - 1];
        std::string   tensor_name          = tensor_to_all_reduce->name;
        // GGML_LOG_INFO("graph compute done for graph number %d, tensor to all reduce: %s, signal: %d\n", graph_number,
        //               tensor_name.c_str(), signal);

        info = info->next;
        try {
            ggml_status status = ggml_backend_graph_compute(backend, graph);
            GGML_ASSERT(status == GGML_STATUS_SUCCESS);
        } catch (const std::exception & e) {
            GGML_LOG_INFO("[%s] exception during graph compute: %s\n", __func__, e.what());
            ggml_free(ctx);
            return false;
        }

        //get the data to be sent first
        std::vector<uint8_t> add_data;
        add_data.resize(ggml_nbytes(tensor_to_all_reduce) + sizeof(tensor_to_all_reduce->name), 0);
        memcpy(add_data.data(), tensor_to_all_reduce->name, sizeof(tensor_to_all_reduce->name));
        ggml_backend_tensor_get(tensor_to_all_reduce, add_data.data() + sizeof(tensor_to_all_reduce->name), 0,
                                ggml_nbytes(tensor_to_all_reduce));
        //now broadcast result to all connected clients
        // GGML_LOG_INFO("Broadcasting all reduce if needed, signal=%d, tensor name: %s", signal, graph->nodes[graph->n_nodes - 1]->name);
        if (signal == 0) {
            //get the tensor to be all reduced

            // GGML_LOG_INFO("doing all reduce for tensor %s\n", tensor_name.c_str());

            //first setup for itself
            block_mutex.lock();
            auto it = all_reduce_blocks.find(tensor_name);
            if (it != all_reduce_blocks.end()) {
                GGML_LOG_INFO("all_reduce block for tensor %s already exists\n", tensor_name.c_str());
                //if the block already exists, just reinit it
                it->second->block_init(tensor_to_all_reduce, signal, device_count, backend, device_id);
            } else {
                GGML_LOG_INFO("creating all_reduce block for tensor %s\n", tensor_name.c_str());
                try {
                    //if not, create the block and add it to the map
                    all_reduce_block * block =
                        new all_reduce_block(tensor_to_all_reduce, signal, device_count, backend);
                    all_reduce_blocks[tensor_name] = block;
                } catch (const std::exception & e) {
                    GGML_LOG_INFO("[%s] error: %s\n", __func__, e.what());
                }
            }
            block_mutex.unlock();

            //notify all other servers to do all_reduce
            // GGML_LOG_INFO("begin notifying all other servers tensor %s\n",tensor_to_all_reduce->name);

            for (auto & sock_weak : sockets_connectto) {
                if (auto sock = sock_weak.second.lock()) {
                    bool status = send_rpc_cmd(sock, RPC_CMD_ALL_REDUCE, add_data.data(), add_data.size(), nullptr, 0);
                    if (!status) {
                        GGML_LOG_INFO("failed to send all_reduce command to %s\n", sock_weak.first.c_str());
                    }
                } else {
                    GGML_LOG_INFO("recreating socket\n");
                    std::string host;
                    int         port;
                    std::string endpoint = sock_weak.first;
                    if (!parse_endpoint(endpoint, host, port)) {
                        GGML_LOG_INFO("unable to parse endpoint %s", endpoint.c_str());
                    }
                    auto socket = socket_connect(host.c_str(), port);
                    if (socket == nullptr) {
                        GGML_LOG_INFO("nullptr socket");
                    }
                    sockets_connectto[endpoint] = socket;
                    // GGML_LOG_INFO("create connection for device %s\n", endpoint.c_str());
                    // GGML_LOG_INFO("sending data to sock %d\n",socket->fd);
                    bool status =
                        send_rpc_cmd(socket, RPC_CMD_ALL_REDUCE, add_data.data(), add_data.size(), nullptr, 0);
                    if (!status) {
                        GGML_LOG_INFO("failed to send all_reduce command to %s\n", sock_weak.first.c_str());
                    }
                }
            }

            //wait for another thread to notify that all_reduce is done
            GGML_LOG_INFO("waiting for all_reduce to complete for tensor %s\n", tensor_name.c_str());
            all_reduce_blocks[tensor_name]->wait_for_completion();

            //uninit the block
            all_reduce_blocks[tensor_name]->block_uinit();
        }
    }

    return true;
}

bool rpc_server::set_split(rpc_msg_set_split_rsp & response) {
    server_split    = true;
    response.result = GGML_STATUS_SUCCESS;
    return true;
}

void rpc_server::add_socket_listen(const std::shared_ptr<socket_t> & sock) {
    std::lock_guard<std::mutex> lock(sockets_mutex);
    //add socket to the list
    sockets_listento.push_back(sock);
    GGML_LOG_INFO("listen to socket %d\n", sock->fd);
}

bool rpc_server::create_peer_connection(const rpc_msg_create_peer_connection_req & request,
                                        rpc_msg_create_peer_connection_rsp &       response) {
    // GGML_LOG_INFO("creating peer connection\n");
    device_id    = request.device_id;
    device_count = request.device_count;
    peer_socks_held.clear();  // re-init: drop any previously held peer sockets
    // Reset stale per-session state so a NEW client run starts clean instead of
    // reusing the previous run's all-reduce blocks / stored graphs (which point at
    // the prior run's freed tensors -> the warmup "Connection closed by peer" crash
    // that otherwise forces an rpc-server restart between every run). Same delete the
    // destructor does; safe here because no compute/all-reduce is in flight yet at
    // connection-setup time.
    {
        std::lock_guard<std::mutex> lock(block_mutex);
        for (auto & block : all_reduce_blocks) {
            delete block.second;
        }
        all_reduce_blocks.clear();
    }
    for (auto & info : graph_compute_infos) {
        delete info.second;
    }
    graph_compute_infos.clear();
#ifdef _WIN32
    {
        WSADATA wsaData;
        int     res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            fprintf(stderr, "WSAStartup failed: %d\n", res);
            return;
        }
    }
#endif
    //for every endpoint that the server receives, connect to it
    for (uint8_t i = 0; i < request.device_count; i++) {
        if (i != device_id) {
            const std::string endpoint = request.endpoints[i];
            auto              it       = sockets_connectto.find(endpoint);
            if (it != sockets_connectto.end()) {
                if (auto sock = it->second.lock()) {
                    GGML_LOG_INFO("already connect");
                }
            }
            std::string host;
            int         port;
            if (!parse_endpoint(endpoint, host, port)) {
                GGML_LOG_INFO("unable to parse endpoint %s", endpoint.c_str());
            }
            auto sock = socket_connect(host.c_str(), port);
            if (sock == nullptr) {
                GGML_LOG_INFO("nullptr socket");
            }
            sockets_connectto[endpoint] = sock;
            peer_socks_held.push_back(sock);  // keep alive so it isn't re-dialed every all-reduce
            GGML_LOG_INFO("create connection for device %s\n", endpoint.c_str());
        }
    }
    response.result = GGML_STATUS_SUCCESS;
    return true;
}

bool rpc_server::all_reduce(std::vector<uint8_t> & input) {
    // GGML_LOG_INFO("receiving all reduce, size: %ld\n",input.size());

    //parse the tensor_name and tensor_data
    char tensor_name_[GGML_MAX_NAME];
    memcpy(tensor_name_, input.data(), sizeof(tensor_name_));
    std::vector<uint8_t> tensor_data;
    tensor_data.resize(input.size() - sizeof(tensor_name_), 0);
    memcpy(tensor_data.data(), input.data() + sizeof(tensor_name_), tensor_data.size());

    // std::string filename=std::to_string(device_id);
    // GGML_LOG_INFO("dumping to file %s\n",filename.c_str());
    // std::ofstream out(filename+".txt", std::ios::app);
    // out << "[" << __func__ << "]" << ", receive all_reduce for tensor " << tensor_name_ << "\n";
    // const float * float_ptr = reinterpret_cast<const float *>(tensor_data.data());
    // for (size_t j = 0; j < tensor_data.size()/sizeof(float); ++j) {
    //     out << static_cast<float>(float_ptr[j]) << " ";
    // }
    // out << "\n";
    // out.close();

    std::string tensor_name = tensor_name_;

    //check whether the matched all reduce block exists
    block_mutex.lock();
    auto it = all_reduce_blocks.find(tensor_name);
    if (it == all_reduce_blocks.end()) {
        // GGML_LOG_INFO("no all_reduce block found for tensor %s, going to buffer it", tensor_name.c_str());
        try {
            all_reduce_block * block = new all_reduce_block();
            block->add_to_buffer(tensor_data);
            all_reduce_blocks[tensor_name] = block;
        } catch (const std::exception & e) {
            GGML_LOG_INFO("[%s] error: %s\n", __func__, e.what());
        }
    } else {
        //if the block exists, check whether it's initialized
        if (it->second->is_init()) {
            //just do the addition
            // GGML_LOG_INFO("all_reduce block for tensor %s found, adding data\n", tensor_name.c_str());
            it->second->add(tensor_data, device_id);

        } else {
            //first buffer it
            // GGML_LOG_INFO("all_reduce block for tensor %s found but not initialized, buffering data\n",
            //   tensor_name.c_str());
            it->second->add_to_buffer(tensor_data);
        }
    }
    block_mutex.unlock();
    return true;
}

rpc_server::~rpc_server() {
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    for (auto block : all_reduce_blocks) {
        delete block.second;
    }
    for (auto info : graph_compute_infos) {
        delete info.second;
    }
}

static void rpc_serve_client(rpc_server & server, sockfd_t sockfd, size_t free_mem, size_t total_mem) {
    // rpc_server server(backend);
    while (true) {
        uint8_t cmd;
        if (!recv_data(sockfd, &cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            fprintf(stderr, "Unknown command: %d\n", cmd);
            break;
        }
        // GGML_LOG_INFO("Received command: %d\n", cmd);
        switch (cmd) {
            case RPC_CMD_ALLOC_BUFFER:
                {
                    rpc_msg_alloc_buffer_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    rpc_msg_alloc_buffer_rsp response;
                    server.alloc_buffer(request, response);
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_GET_ALLOC_SIZE:
                {
                    rpc_msg_get_alloc_size_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    rpc_msg_get_alloc_size_rsp response;
                    server.get_alloc_size(request, response);
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_GET_ALIGNMENT:
                {
                    if (!recv_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    rpc_msg_get_alignment_rsp response;
                    server.get_alignment(response);
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_GET_MAX_SIZE:
                {
                    if (!recv_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    rpc_msg_get_max_size_rsp response;
                    server.get_max_size(response);
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_BUFFER_GET_BASE:
                {
                    rpc_msg_buffer_get_base_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    rpc_msg_buffer_get_base_rsp response;
                    if (!server.buffer_get_base(request, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_FREE_BUFFER:
                {
                    rpc_msg_free_buffer_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    if (!server.free_buffer(request)) {
                        return;
                    }
                    if (!send_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_BUFFER_CLEAR:
                {
                    rpc_msg_buffer_clear_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    if (!server.buffer_clear(request)) {
                        return;
                    }
                    if (!send_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_SET_TENSOR:
                {
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    if (!server.set_tensor(input)) {
                        return;
                    }
                    if (!send_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_INIT_TENSOR:
                {
                    rpc_msg_init_tensor_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    if (!server.init_tensor(request)) {
                        return;
                    }
                    if (!send_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_GET_TENSOR:
                {
                    rpc_msg_get_tensor_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    std::vector<uint8_t> response;
                    if (!server.get_tensor(request, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, response.data(), response.size())) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_COPY_TENSOR:
                {
                    rpc_msg_copy_tensor_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    rpc_msg_copy_tensor_rsp response;
                    if (!server.copy_tensor(request, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_GRAPH_COMPUTE:
                {
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    rpc_msg_graph_compute_rsp response;
                    if (!server.graph_compute(input, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_GET_DEVICE_MEMORY:
                {
                    if (!recv_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    rpc_msg_get_device_memory_rsp response;
                    response.free_mem  = free_mem;
                    response.total_mem = total_mem;
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_SET_SPLIT:
                {
                    if (!recv_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    rpc_msg_set_split_rsp response;
                    if (!server.set_split(response)) {
                        return;
                    }
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_CREATE_PEER_CONNECTION:
                {
                    rpc_msg_create_peer_connection_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    rpc_msg_create_peer_connection_rsp response;
                    if (!server.create_peer_connection(request, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_ALL_REDUCE:
                {
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    if (!server.all_reduce(input)) {
                        return;
                    }
                    if (!send_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_DO_COMPUTATION:
                {
                    rpc_msg_do_computation_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    if (!server.do_computation(request)) {
                        return;
                    }
                    if (!send_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_LOAD_CACHED:
                {
                    rpc_msg_load_cached_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    rpc_msg_load_cached_rsp response;
                    if (!server.load_cached(request, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_SET_TENSOR_CACHE:
                {
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    if (!server.set_tensor_cache(input)) {
                        return;
                    }
                    if (!send_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    break;
                }
            default:
                {
                    fprintf(stderr, "Unknown command: %d\n", cmd);
                    return;
                }
        }
    }
}

static void get_backend_memory(size_t * free_mem, size_t * total_mem) {
#ifdef GGML_USE_CUDA
    ggml_backend_cuda_get_device_memory(0, free_mem, total_mem);
#elif GGML_USE_VULKAN
    ggml_backend_vk_get_device_memory(0, free_mem, total_mem);
#elif GGML_USE_SYCL
    ggml_backend_sycl_get_device_memory(0, free_mem, total_mem);
#else
#    ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total_mem = status.ullTotalPhys;
    *free_mem  = status.ullAvailPhys;
#    else
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total_mem     = pages * page_size;
    *free_mem      = *total_mem;
#    endif
#endif
}

void ggml_backend_rpc_start_server(ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem) {
    std::string host;
    int         port;
    rpc_server  server(backend);
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }
#ifdef _WIN32
    {
        WSADATA wsaData;
        int     res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            fprintf(stderr, "WSAStartup failed: %d\n", res);
            return;
        }
    }
#endif
    auto server_socket = create_server_socket(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        printf("waiting for socket\n");
        auto client_socket = socket_accept(server_socket->fd);
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        //record the socket accept
        server.add_socket_listen(client_socket);

        //for every socket accept, create a thread to trace it
        std::thread([client_socket, &server] {
            size_t free_mem, total_mem;
            get_backend_memory(&free_mem, &total_mem);
            printf("Accepted client connection, free_mem=%zu, total_mem=%zu\n", free_mem, total_mem);
            printf("client socket: %d\n", client_socket->fd);
            fflush(stdout);
            rpc_serve_client(server, client_socket->fd, free_mem, total_mem);
            printf("Client connection closed\n\n");
            fflush(stdout);
        }).detach();
    }
#ifdef _WIN32
    WSACleanup();
#endif
    GGML_UNUSED(free_mem);
    GGML_UNUSED(total_mem);
}

// device interface

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *) dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *) dev->context;

    return ctx->name.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *) dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), free, total);

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *) dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str());

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *) dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str());

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *) dev->context;
    // split buffers can only be used with GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_rpc_split(op->src[i]->buffer->buft)) {
                return false;
            }
        }
    }
    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_rpc(op->src[i]->buffer->buft)) {
            ggml_backend_rpc_buffer_type_context * buft_ctx =
                (ggml_backend_rpc_buffer_type_context *) op->src[i]->buffer->buft->context;
            if (buft_ctx->endpoint != dev_ctx->endpoint) {
                return false;
            }
        }
    }
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                // struct ggml_tensor * b = op->src[1];
                // for small weight matrices the active device can end up without any rows, don't use row split in those cases
                // this avoids some edge cases (and the performance would not be good anyways)
                if (a->buffer && ggml_backend_buft_is_rpc_split(a->buffer->buft)) {
                    ggml_backend_rpc_split_buffer_type_context * buft_ctx =
                        (ggml_backend_rpc_split_buffer_type_context *) a->buffer->buft->context;
                    int64_t row_low;
                    int64_t row_high;
                    rpc_get_row_split(&row_low, &row_high, a, buft_ctx->tensor_split,
                                      ggml_backend_rpc_get_device_id(dev_ctx->endpoint.c_str()));
                    if (row_low == row_high) {
                        return false;
                    }
                }
            }
        default:
            break;
    }
    //TODO: call the remote backend and cache the results

    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || (buft->iface.get_name != ggml_backend_rpc_buffer_type_name &&
                  buft->iface.get_name != ggml_backend_rpc_split_buffer_type_name)) {
        return false;
    }
    if (buft->iface.get_name == ggml_backend_rpc_split_buffer_type_name) {
        ggml_backend_rpc_split_buffer_type_context * buft_ctx =
            (ggml_backend_rpc_split_buffer_type_context *) buft->context;
        ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *) dev->context;
        return buft_ctx->endpoint == dev_ctx->endpoint;
    } else {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *) buft->context;
        ggml_backend_rpc_device_context *      dev_ctx  = (ggml_backend_rpc_device_context *) dev->context;
        return buft_ctx->endpoint == dev_ctx->endpoint;
    }
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    return "RPC";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static bool ggml_backend_rpc_create_peer_connection() {
    rpc_msg_create_peer_connection_req request;
    int                                device_count = ggml_backend_rpc_get_device_count();
    request.device_count                            = device_count;
    memset(request.endpoints, 0, sizeof(request.endpoints));
    for (int id = 0; id < device_count; ++id) {
        ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
        strcpy(request.endpoints[id], dev_ctx->endpoint.c_str());
    }
    std::vector<std::thread> threads;
    for (int id = 0; id < device_count; ++id) {
        threads.emplace_back([&, id]() {
            rpc_msg_create_peer_connection_req device_req;
            memcpy(&device_req, &request, sizeof(rpc_msg_create_peer_connection_req));
            device_req.device_id = id;
            auto dev_ctx         = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
            rpc_msg_create_peer_connection_rsp response;
            bool status = send_rpc_cmd(get_socket(dev_ctx->endpoint), RPC_CMD_CREATE_PEER_CONNECTION, &device_req,
                                       sizeof(device_req), &response, sizeof(response));
            GGML_ASSERT(status);
        });
    }
    for (auto & thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    return true;
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_device") == 0) {
        return (void *) ggml_backend_rpc_add_device;
    }
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *) ggml_backend_rpc_split_buffer_type;
    }
    if (strcmp(name, "ggml_backend_rpc_create_peer_connection") == 0) {
        return (void *) ggml_backend_rpc_create_peer_connection;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ reg_ctx,
    };

    return &ggml_backend_rpc_reg;
}

ggml_backend_dev_t ggml_backend_rpc_add_device(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_dev_t> dev_map;
    static std::mutex                                          mutex;
    std::lock_guard<std::mutex>                                lock(mutex);

    if (dev_map.find(endpoint) != dev_map.end()) {
        return dev_map[endpoint];
    }

    ggml_backend_rpc_device_context * ctx = new ggml_backend_rpc_device_context{
        /* .endpoint = */ endpoint,
        /* .name     = */ "RPC[" + std::string(endpoint) + "]",
    };

    ggml_backend_dev_t dev = new ggml_backend_device{
        /* .iface   = */ ggml_backend_rpc_device_i,
        /* .reg     = */ ggml_backend_rpc_reg(),
        /* .context = */ ctx,
    };

    dev_map[endpoint]                      = dev;
    ggml_backend_rpc_reg_context * reg_ctx = (ggml_backend_rpc_reg_context *) dev->reg->context;
    reg_ctx->devices.push_back(dev);
    return dev;
}

GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
