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
#include <filesystem>
#ifndef _WIN32
#    include <utime.h>
#endif
#include <chrono>
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
    sockfd_t   fd;
    // Serializes a whole request/response (or a one-way send) on this socket. send_rpc_cmd
    // does 3 sends + 2 recvs on one fd and is NOT atomic, so two threads issuing RPCs on the
    // SAME socket (e.g. the first forward's threaded graph-send calling BUFFER_GET_BASE while
    // the threaded weight-upload tail still runs SET_TENSOR) interleave their bytes and one
    // call reads ANOTHER call's response -> a buffer caches the WRONG base -> the intermittent
    // warmup deserialize OOB. This is a socket byte-stream race (syscalls), invisible to TSan.
    std::mutex io_mtx;

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
    RPC_CMD_LOAD_CACHED,          // "do you have this slice cached? if so load it into the buffer" (skip upload)
    RPC_CMD_SET_TENSOR_CACHE,     // set_tensor that ALSO persists the slice to the on-disk weight cache
    RPC_CMD_GRAPH_COMPUTE_BATCH,  // all of a token's segment-graphs in ONE round-trip (vs one RPC per segment)
    RPC_CMD_PATCH_VIEWS,          // diff cache: patch only the per-token-changed tensors of an already-stored graph
    RPC_CMD_GRAPH_ADVANCE,  // (#3 prefetch=eliminate) advance the last-patched tensors by their cached stride (no payload bytes)
    RPC_CMD_AR_RESULT,  // tree all-reduce: root -> non-roots, the final folded result (seq|name|f32 data)
    RPC_CMD_GRAPH_COMPUTE_STORE,  // (PP diff cache) non-split: deserialize + STORE + compute inline (MISS)
    RPC_CMD_PATCH_COMPUTE,        // (PP diff cache) non-split: patch the stored graph + compute inline (HIT)
    RPC_CMD_ADVANCE_COMPUTE,  // (PP diff cache + prefetch) non-split: advance stored graph by cached stride + compute inline (predicted HIT, no patch payload)
    RPC_CMD_SEND_TO_PEER,  // direct pipeline handoff: the SRC server pushes a tensor straight into the DST peer's buffer (bypasses the client relay)
    RPC_CMD_BATCH_LOAD_CACHED,  // batched warm load: one message lists all (tensor,key) a server owns; it loads every hit from its LOCAL cache (no per-tensor round-trip) and returns a hit byte per entry
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

// Direct pipeline handoff (RPC_CMD_SEND_TO_PEER). The client tells the SRC server to
// push `src` straight into the DST peer's buffer described by `dst` (over the SRC
// server's existing peer connection to `dst_endpoint`), instead of relaying the bytes
// src -> client -> dst. Carries no payload: the SRC server reads its own local bytes.
struct rpc_msg_send_to_peer_req {
    rpc_tensor src;                 // locate the bytes on the SOURCE server
    rpc_tensor dst;                 // where to write them on the DESTINATION server
    char       dst_endpoint[256];   // which peer to push to (key into sockets_connectto)
};

struct rpc_msg_send_to_peer_rsp {
    uint8_t result;   // 1 = pushed+acked by the peer; 0 = fall back to the client relay
};

struct rpc_msg_do_computation_req {
    uint8_t graph_number;
};

// Diff cache (RPC_CMD_PATCH_VIEWS). When a decode token's graph is structurally
// identical to one the servers already hold, we don't re-ship the ~1000-tensor
// structure -- we ship only the per-token-changed tensors. Per token the per-device
// value-fields advance (KV-cache write positions in data/view_offs, position state in
// op_params, and -- with tensor-parallel "k"-cache splits -- the split-adjusted ne/nb).
// A patch carries the changed tensor's full rpc_tensor; the server copies the value
// fields (ne/nb/op_params/flags/data/view_offs) onto its stored ggml_tensor, keeping
// the stored wiring (src/view_src/buffer) intact. The tensor is addressed by its index
// in the segment's originally-sent tensor array -- the server records that order at
// store time so client and server agree on the index. We deliberately do NOT diff the
// id/src/view_src/buffer/name fields: those are client pointers that change every token
// but the server's stored graph already has the correct (fixed) wiring.
// Payload (per device): graph_number(1) | n_segments(4) |
//   per segment: n_patches(4) | rpc_view_patch[n_patches]   (segments in chain order)
struct rpc_view_patch {
    uint32_t   idx;
    rpc_tensor t;
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
    // atomic so the lock-free fast path in get_base (read) doesn't data-race the cache store
    // (which happens under g_get_base_mutex). acquire/release pair the read with that store.
    std::atomic<void *>       base_ptr;
    uint64_t                  remote_ptr;
};

struct ggml_tensor_extra_rpc {
    struct ggml_backend_rpc_buffer_context * buffer_ctx[RPC_MAX_DEVICES];
    //maybe we don't need to store the rows
    std::pair<int64_t, int64_t>              rows[RPC_MAX_DEVICES];
    int                                      split_dim = -1;
    // (activation pool) gallocr byte offset of this tensor within its compute buffer. When a
    // replicated activation lives in a shared per-device pool buffer (RPC_POOL), the server
    // address is pool_base + data_off (the pool reuses one buffer for all activations via gallocr's
    // already-computed reuse plan, instead of one server buffer per activation). 0 for non-pooled.
    uint64_t                                 data_off = 0;
};

static bool rpc_opt_enabled();  // fwd decl (defined below) -- persist rides the default optimized path
// (#3d) reuse a split tensor's server buffer ACROSS tokens (DEFAULT ON; opt out: RPC_NO_PERSIST_BUFFERS)
// instead of re-allocating one per token. init_tensor currently does a per-tensor RPC_CMD_ALLOC_BUFFER for
// every activation every token (~1500/token), and each fresh remote_ptr then needs a get_base RPC
// (~1500/token) -- ~3000 WiFi round-trips/token, the dominant decode-build cost. The cgraph reuses
// pool addresses each token, so the same (tensor-ptr, device) recurs with the same size = same
// logical tensor = safe to reuse its buffer (and the reused ctx keeps its cached base, killing the
// get_base storm too). Persisted ctxs outlive the per-token extras; the per-token free path skips
// them (cache owns them) and they're released when replaced (size change) or at process exit
// (bounded by the unique-tensor set).
static const bool g_persist_buffers = rpc_opt_enabled() && getenv("RPC_NO_PERSIST_BUFFERS") == nullptr;
static std::mutex g_persist_mtx;
static std::map<std::pair<const void *, int>, std::pair<ggml_backend_rpc_buffer_context *, size_t>> g_persist_buf;
static std::unordered_set<const void *>                                                             g_persist_ctxs;

// (activation pool, DEFAULT ON; opt out RPC_NO_POOL) Cross-layer sub-allocation. Persist (above) allocated ONE
// server buffer per activation (~1500/token) -- fine for SPEED (reused across tokens) but each device
// then holds EVERY layer's activations at once (~18 GB for a long prefill), because the per-tensor
// alloc bypasses gallocr's cross-layer reuse. gallocr already computed a reuse plan: the compute
// buffer's size is the PEAK (~280 MB) and each tensor's gallocr offset (in tensor->data) reuses space
// across non-overlapping tensors. So: allocate ONE pool per (compute buffer, device) sized at the
// gallocr peak, and point every replicated activation at pool_base + its gallocr offset (extra->
// data_off). Cuts each replicated device from the all-layer SUM to the peak (~64x). Pool is reused
// across tokens (resized only if the buffer grows) and freed when the compute buffer is freed.
static const bool g_act_pool_on = rpc_opt_enabled() && getenv("RPC_NO_POOL") == nullptr;
static std::mutex g_act_pool_mtx;
static std::map<std::pair<const void *, int>, std::pair<ggml_backend_rpc_buffer_context *, size_t>> g_act_pool;

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
    // hold the socket for the WHOLE exchange so a concurrent RPC on the same fd can't
    // interleave its bytes and cross responses (see socket_t::io_mtx).
    std::lock_guard<std::mutex> io_lock(sock->io_mtx);
    uint8_t                     cmd_byte = cmd;
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

// Fire-and-forget command: send cmd|size|payload and DON'T wait for a reply. Used for
// the peer-to-peer all-reduce broadcast, where the app-level ack added a full round-trip
// per peer per reduce but no reliability (TCP already guarantees delivery+order, the
// receiver buffers partials by seq, and a dead link is caught by the all-reduce timeout).
// The matching server handler must NOT send a reply on this socket, or the unread acks
// would pile up in the sender's recv buffer and eventually backpressure the peer.
//
// Latency: pack cmd|size|payload into ONE contiguous buffer and a single send_data, so the
// broadcast is one segment on the wire instead of up to three (with TCP_NODELAY each
// separate send() can be its own packet on the latency-bound shared-medium cell -- this is
// the all-reduce hot path, ~44 reduces/token x (N-1) peers).
static bool send_rpc_cmd_oneway(const std::shared_ptr<socket_t> & sock, enum rpc_cmd cmd, const void * input,
                                size_t input_size) {
    if (sock == nullptr) {
        GGML_LOG_INFO("[send_rpc_cmd_oneway] NULL socket for cmd %d\n", (int) cmd);
        return false;
    }
    uint8_t              cmd_byte = cmd;
    std::vector<uint8_t> framed(sizeof(cmd_byte) + sizeof(input_size) + input_size);
    memcpy(framed.data(), &cmd_byte, sizeof(cmd_byte));
    memcpy(framed.data() + sizeof(cmd_byte), &input_size, sizeof(input_size));
    if (input_size > 0) {
        memcpy(framed.data() + sizeof(cmd_byte) + sizeof(input_size), input, input_size);
    }
    // one framed write, but still serialize vs other senders on this fd (see socket_t::io_mtx)
    std::lock_guard<std::mutex> io_lock(sock->io_mtx);
    return send_data(sock->fd, framed.data(), framed.size());
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
    // (activation pool) release this compute buffer's per-device pools (other devices).
    if (g_act_pool_on) {
        std::lock_guard<std::mutex> lock(g_act_pool_mtx);
        for (auto it = g_act_pool.begin(); it != g_act_pool.end();) {
            if (it->first.first == (const void *) buffer) {
                rpc_msg_free_buffer_req freq = { it->second.first->remote_ptr };
                send_rpc_cmd(it->second.first->sock, RPC_CMD_FREE_BUFFER, &freq, sizeof(freq), nullptr, 0);
                delete it->second.first;
                it = g_act_pool.erase(it);
            } else {
                ++it;
            }
        }
    }
    delete ctx;
}

// BUFFER_GET_BASE is a lazy, cached RPC. The threaded buffer-alloc/upload loops call these
// concurrently, and ctx's for one device share a socket; without serialization two in-flight
// BUFFER_GET_BASE on the same socket interleave their responses and one ctx caches ANOTHER
// buffer's base -> later a tensor's data points into the wrong buffer -> the intermittent
// deserialize OOB (GGML_ASSERT(tensor->data >= buffer_start)) at warmup. Serialize the fetch
// (double-checked, so the cached fast path stays lock-free).
static std::mutex g_get_base_mutex;

// Fetch + cache the buffer base once, under g_get_base_mutex so concurrent first-calls from
// the threaded alloc/upload loops can't interleave BUFFER_GET_BASE responses on a shared
// socket (which cached the WRONG buffer's base -> the intermittent deserialize OOB at warmup).
// The fast path is a lock-free atomic acquire-load that pairs with the release-store below.
static void * rpc_get_base_cached(ggml_backend_rpc_buffer_context * ctx) {
    void * p = ctx->base_ptr.load(std::memory_order_acquire);
    if (p != nullptr) {
        return p;
    }
    std::lock_guard<std::mutex> lock(g_get_base_mutex);
    p = ctx->base_ptr.load(std::memory_order_acquire);
    if (p != nullptr) {
        return p;
    }
    rpc_msg_buffer_get_base_req request = { ctx->remote_ptr };
    rpc_msg_buffer_get_base_rsp response;
    bool                        status =
        send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    GGML_ASSERT(status);
    p = reinterpret_cast<void *>(response.base_ptr);
    ctx->base_ptr.store(p, std::memory_order_release);
    static const bool dbg_getbase = getenv("RPC_DBG_GETBASE") != nullptr;
    if (dbg_getbase) {
        GGML_LOG_INFO("[GETBASE] ctx=%p remote_ptr=0x%llx -> base=%p (fd=%d)\n", (void *) ctx,
                      (unsigned long long) ctx->remote_ptr, p, ctx->sock ? ctx->sock->fd : -1);
    }
    return p;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    return rpc_get_base_cached((ggml_backend_rpc_buffer_context *) buffer->context);
}

static void * ggml_backend_rpc_buffer_context_get_base(ggml_backend_rpc_buffer_context * ctx) {
    return rpc_get_base_cached(ctx);
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
                    // (#3d) a persisted buffer is owned by g_persist_buf and shared across this and
                    // future tokens' extras -- don't free or delete it here (would be a use-after-
                    // free next token); the cache releases it on size-change or at exit.
                    bool persisted = false;
                    if (g_persist_buffers) {
                        std::lock_guard<std::mutex> lock(g_persist_mtx);
                        persisted = g_persist_ctxs.count(ctx_item[i]) != 0;
                    }
                    if (persisted) {
                        continue;
                    }
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
    ggml_backend_rpc_buffer_context * ctx =
        extra ? (ggml_backend_rpc_buffer_context *) extra->buffer_ctx[device_id] : nullptr;
    if (ctx) {
        result.buffer = ctx->remote_ptr;  //use remote_ptr to find the exact buffer on the remote server
        result.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(ctx));
    } else {
        // A tensor can reach the split buffer with no `extra`, or with no per-device
        // buffer_ctx[device_id]. The old code logged this but then unconditionally
        // dereferenced `extra->split_dim` below -> SIGSEGV during model load (this is
        // the llama-bench -sm row crash). Fall through to the non-split path instead.
        result.buffer = 0;
        result.data   = 0;
        GGML_LOG_INFO("Error: tensor %s missing %s for split; serializing non-split\n",
                      tensor->name, extra ? "device buffer_ctx" : "extra");
    }
    int split_dim = extra ? extra->split_dim : -1;
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
        // non-split (or extra-less) tensor: copy dimensions straight through so the
        // remote gets a valid descriptor instead of uninitialized ne/nb.
        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            result.ne[i] = tensor->ne[i];
            result.nb[i] = tensor->nb[i];
        }
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

// Master optimization gate for the research-paper A/B. RPC_NO_OPT set => ALL of our
// optimizations are OFF (faithful baseline); unset => all ON (the optimized path). Read
// once. Must be set consistently on the client AND every rpc-server for the all-reduce
// fire-and-forget vs acked paths to agree. Covers: diff cache, graph-send batching,
// parallel/fire-and-forget all-reduce broadcast + deterministic ordered fold (and 1A/1B/1C),
// the on-disk weight cache, and the threaded weight upload/alloc/download. Finer-grained
// gates (RPC_GRAPH_CACHE, RPC_NO_WEIGHT_CACHE, RPC_SERIAL_UPLOAD) still apply on top.
static bool rpc_opt_enabled() {
    static const bool on = (getenv("RPC_NO_OPT") == nullptr);
    return on;
}

// (#3) graph-send fire-and-forget (opt-in, RPC_GRAPH_ONEWAY): send the per-token PATCH_VIEWS /
// GRAPH_ADVANCE as a one-way message (no ack) pipelined before DO_COMPUTATION on the same ordered
// socket, removing the ack round-trip. Must be set consistently on client AND every server: when
// on, the server also SKIPS the ack for those commands (else its ack bytes would pollute the next
// DO_COMPUTATION response on the shared fd). TCP reliable+ordered => the patch is applied before
// compute (same guarantee the all-reduce fire-and-forget broadcast already relies on).
static bool rpc_graph_oneway() {
    // DEFAULT ON under the optimized path; opt out with RPC_NO_GRAPH_ONEWAY (byte-identical either way).
    static const bool on = rpc_opt_enabled() && getenv("RPC_NO_GRAPH_ONEWAY") == nullptr;
    return on;
}

// (#6) all-reduce partial wire format (opt-in, RPC_AR_PARTIAL=fp16|int8|f32), generalizing the
// fp16 path: ship each f32 partial narrowed (f16 = half the bytes, int8 = a quarter + a per-
// partial scale), convert back to f32 on receive so the fold stays f32 + id-ordered
// (deterministic). NOT bit-identical to the f32 path (each partial is rounded/quantized) -- a
// quality/throughput tradeoff, off by default, must be set consistently on every node. int8
// uses symmetric per-partial scaling (scale = maxabs/127), shipped as: scale(f32) | int8[n].
enum class rpc_ar_fmt { f32, f16, i8, i8b, e4m3 };

static rpc_ar_fmt rpc_ar_partial() {
    static const rpc_ar_fmt fmt = []() -> rpc_ar_fmt {
        if (!rpc_opt_enabled()) {
            return rpc_ar_fmt::f32;
        }
        if (const char * p = getenv("RPC_AR_PARTIAL")) {
            if (strcmp(p, "int8b") == 0 || strcmp(p, "i8b") == 0 || strcmp(p, "q8") == 0) {
                return rpc_ar_fmt::i8b;
            }
            if (strcmp(p, "int8") == 0 || strcmp(p, "i8") == 0) {
                return rpc_ar_fmt::i8;
            }
            if (strcmp(p, "fp8") == 0 || strcmp(p, "e4m3") == 0) {
                return rpc_ar_fmt::e4m3;
            }
            if (strcmp(p, "fp16") == 0 || strcmp(p, "f16") == 0) {
                return rpc_ar_fmt::f16;
            }
            if (strcmp(p, "f32") == 0 || strcmp(p, "none") == 0) {
                return rpc_ar_fmt::f32;
            }
        }
        return getenv("RPC_AR_FP16") != nullptr ? rpc_ar_fmt::f16 : rpc_ar_fmt::f32;  // back-compat
    }();
    return fmt;
}

// f32 -> OCP fp8 e4m3 (1 sign, 4 exp bias 7, 3 mantissa; max 448, no inf). Round-to-nearest-even;
// |x|>448 -> 448; |x| below the smallest normal (2^-6) flushes to 0 (subnormals not emitted).
static inline uint8_t rpc_f32_to_e4m3(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint8_t  sign = (uint8_t) ((x >> 24) & 0x80);
    const uint32_t absx = x & 0x7FFFFFFF;
    if (absx >= 0x7F800000) {
        return (uint8_t) (sign | 0x7E);
    }  // inf/nan -> max finite 448
    int32_t e4 = (int32_t) (absx >> 23) - 127 + 7;  // target biased exponent
    if (e4 <= 0) {
        return sign;
    }  // underflow -> +/-0
    uint32_t m   = absx & 0x7FFFFF;
    uint32_t m3  = m >> 20;      // top 3 mantissa bits
    uint32_t rem = m & 0xFFFFF;  // round-to-nearest-even on the rest
    if (rem > 0x80000 || (rem == 0x80000 && (m3 & 1))) {
        if (++m3 == 8) {
            m3 = 0;
            e4++;
        }
    }
    if (e4 >= 16 || (e4 == 15 && m3 >= 7)) {
        return (uint8_t) (sign | 0x7E);
    }  // overflow / NaN slot -> 448
    return (uint8_t) (sign | (e4 << 3) | m3);
}

static inline float rpc_e4m3_to_f32(uint8_t v) {
    const uint32_t sign = (uint32_t) (v & 0x80) << 24;
    const uint32_t e4   = (v >> 3) & 0xF;
    const uint32_t m3   = v & 0x7;
    uint32_t       bits = sign;  // e4==0 => +/-0 (we never emit subnormals)
    if (e4 != 0) {
        bits |= ((e4 - 7 + 127) << 23) | (m3 << 20);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void rpc_e4m3_quantize(const float * x, uint8_t * q, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        q[i] = rpc_f32_to_e4m3(x[i]);
    }
}

static void rpc_e4m3_dequantize(const uint8_t * q, float * x, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        x[i] = rpc_e4m3_to_f32(q[i]);
    }
}

// symmetric per-partial int8: returns the scale; q[i] = clamp(round(x[i]/scale), -127, 127)
static float rpc_i8_quantize(const float * x, int8_t * q, int64_t n) {
    float maxabs = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        maxabs = std::max(maxabs, std::fabs(x[i]));
    }
    const float scale = maxabs > 0.0f ? maxabs / 127.0f : 1.0f;
    const float inv   = 1.0f / scale;
    for (int64_t i = 0; i < n; i++) {
        int v = (int) lrintf(x[i] * inv);
        q[i]  = (int8_t) std::max(-127, std::min(127, v));
    }
    return scale;
}

static void rpc_i8_dequantize(const int8_t * q, float scale, float * x, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        x[i] = (float) q[i] * scale;
    }
}

// per-block int8 (Q8_0-style, RPC_AR_PARTIAL=int8b): split the partial into blocks of 32, each
// with its OWN f16 scale (= block-maxabs/127). Far more accurate than a single scale over the
// whole ~2048-elem partial (each block adapts to its local magnitude) -> the usable 1-byte
// format. Wire layout per block: ggml_fp16_t scale | int8 q[len]. ~1.06 bytes/elem.
static const int64_t RPC_Q8_BLK = 32;

static size_t rpc_i8b_bytes(int64_t n) {
    const int64_t nblk = (n + RPC_Q8_BLK - 1) / RPC_Q8_BLK;
    return (size_t) nblk * sizeof(ggml_fp16_t) + (size_t) n * sizeof(int8_t);
}

static void rpc_i8b_quantize(const float * x, uint8_t * out, int64_t n) {
    size_t off = 0;
    for (int64_t b = 0; b < n; b += RPC_Q8_BLK) {
        const int64_t len    = std::min<int64_t>(RPC_Q8_BLK, n - b);
        float         maxabs = 0.0f;
        for (int64_t i = 0; i < len; i++) {
            maxabs = std::max(maxabs, std::fabs(x[b + i]));
        }
        const float scale = maxabs > 0.0f ? maxabs / 127.0f : 1.0f;
        const float inv   = maxabs > 0.0f ? 127.0f / maxabs : 0.0f;
        ggml_fp16_t hs;
        ggml_fp32_to_fp16_row(&scale, &hs, 1);
        memcpy(out + off, &hs, sizeof(hs));
        off += sizeof(hs);
        for (int64_t i = 0; i < len; i++) {
            const int v = (int) lrintf(x[b + i] * inv);
            out[off++]  = (uint8_t) (int8_t) std::max(-127, std::min(127, v));
        }
    }
}

static void rpc_i8b_dequantize(const uint8_t * in, float * x, int64_t n) {
    size_t off = 0;
    for (int64_t b = 0; b < n; b += RPC_Q8_BLK) {
        const int64_t len = std::min<int64_t>(RPC_Q8_BLK, n - b);
        ggml_fp16_t   hs;
        memcpy(&hs, in + off, sizeof(hs));
        off += sizeof(hs);
        float scale;
        ggml_fp16_to_fp32_row(&hs, &scale, 1);
        for (int64_t i = 0; i < len; i++) {
            const int8_t q = (int8_t) in[off++];
            x[b + i]       = (float) q * scale;
        }
    }
}

// bytes a narrowed f32 partial of n elements occupies on the wire (f16: 2n; int8: 4 + n; fp8: n)
static size_t rpc_ar_payload_bytes(rpc_ar_fmt fmt, int64_t nelem, size_t nbytes_f32) {
    switch (fmt) {
        case rpc_ar_fmt::f16:
            return (size_t) nelem * sizeof(ggml_fp16_t);
        case rpc_ar_fmt::i8:
            return sizeof(float) + (size_t) nelem * sizeof(int8_t);
        case rpc_ar_fmt::i8b:
            return rpc_i8b_bytes(nelem);
        case rpc_ar_fmt::e4m3:
            return (size_t) nelem * sizeof(uint8_t);
        default:
            return nbytes_f32;
    }
}

// TREE all-reduce (opt-in, RPC_AR_TREE): instead of all-to-all (every server broadcasts its
// partial to all peers => N(N-1) transmissions/reduce), non-root servers send their partial
// to ROOT (device 0) only; root folds all N in ascending-device-id order (REUSES the slot
// fold, so the result is bit-identical to all-to-all) then broadcasts the single f32 result
// to the N-1 non-roots => 2(N-1) transmissions, halving shared-medium airtime/contention.
// Result is broadcast as f32 (exact) even under fp16 partials, so every server ends with the
// root's identical result. Must be set consistently on all rpc-servers.
static bool rpc_ar_tree() {
    static const bool on = rpc_opt_enabled() && (getenv("RPC_AR_TREE") != nullptr);
    return on;
}

// (activation pool) get-or-create the per-device pool buffer for a compute `buffer`, sized >= `size`.
// One pool per (compute buffer, device); reused across tokens, grown (free + realloc) if needed.
static ggml_backend_rpc_buffer_context * rpc_get_act_pool(const void * buffer, int id, size_t size,
                                                          const std::shared_ptr<socket_t> & sock) {
    std::lock_guard<std::mutex> lock(g_act_pool_mtx);
    auto                        key = std::make_pair(buffer, id);
    auto                        it  = g_act_pool.find(key);
    if (it != g_act_pool.end()) {
        if (it->second.second >= size) {
            return it->second.first;
        }
        rpc_msg_free_buffer_req freq = { it->second.first->remote_ptr };
        send_rpc_cmd(it->second.first->sock, RPC_CMD_FREE_BUFFER, &freq, sizeof(freq), nullptr, 0);
        delete it->second.first;
        g_act_pool.erase(it);
    }
    rpc_msg_alloc_buffer_req req = { size };
    rpc_msg_alloc_buffer_rsp rsp;
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &req, sizeof(req), &rsp, sizeof(rsp));
    GGML_ASSERT(status);
    if (rsp.remote_ptr == 0) {
        return nullptr;
    }
    auto * ctx      = new ggml_backend_rpc_buffer_context{ sock, nullptr, rsp.remote_ptr };
    g_act_pool[key] = { ctx, size };
    return ctx;
}

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

            // (activation pool) record gallocr's byte offset of this tensor within the compute buffer
            // (tensor->data is base+offset; views/cache keep their own buffer so they're skipped).
            if (g_act_pool_on && !found && !cache) {
                extra->data_off = (uint64_t) tensor->data - (uint64_t) ggml_backend_rpc_buffer_context_get_base(ctx);
            }

            //allocate buffer on other servers
            // Allocate each device's buffer concurrently: extra->buffer_ctx[id]/rows[id]
            // are per-id (disjoint) writes, sockets are pre-fetched + held so the weak_ptr
            // cache can't churn under the workers. RPC_SERIAL_UPLOAD=1 reverts to serial.
            // (On alloc failure we set buffer_ctx[id]=nullptr -- which downstream already
            //  null-checks -- instead of `delete extra`, which under threading would
            //  double-free and was a use-after-free even serially via `tensor->extra=extra`.)
            static const bool serial_alloc = (!rpc_opt_enabled() || getenv("RPC_SERIAL_UPLOAD") != nullptr);
            const int         n_dev        = ggml_backend_rpc_get_device_count();
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
                    // (activation pool) replicated activations share ONE per-device pool buffer at
                    // gallocr offsets instead of one buffer each -> all-layer working set drops to the
                    // gallocr peak. cache tensors keep their own buffer (not gallocr-managed the same).
                    if (g_act_pool_on && !cache) {
                        extra->buffer_ctx[id] = rpc_get_act_pool((const void *) buffer, id, buffer->size, socks[id]);
                        extra->rows[id]       = { 0, tensor->ne[0] };
                        return;
                    }
                    // (#3d) persist: reuse this (tensor,id)'s server buffer from a prior token if
                    // the size still matches -> skip the ALLOC_BUFFER RPC (and the later get_base,
                    // since the reused ctx keeps its cached base). A size change frees + replaces it.
                    if (g_persist_buffers) {
                        std::lock_guard<std::mutex> lock(g_persist_mtx);
                        auto                        pkey = std::make_pair((const void *) tensor, id);
                        auto                        pit  = g_persist_buf.find(pkey);
                        if (pit != g_persist_buf.end()) {
                            if (pit->second.second == size) {
                                extra->buffer_ctx[id] = pit->second.first;
                                extra->rows[id]       = { 0, tensor->ne[0] };
                                return;  // reuse -- no ALLOC_BUFFER round-trip
                            }
                            rpc_msg_free_buffer_req freq = { pit->second.first->remote_ptr };
                            send_rpc_cmd(pit->second.first->sock, RPC_CMD_FREE_BUFFER, &freq, sizeof(freq), nullptr, 0);
                            g_persist_ctxs.erase(pit->second.first);
                            delete pit->second.first;
                            g_persist_buf.erase(pit);
                        }
                    }
                    bool status = send_rpc_cmd(socks[id], RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response,
                                                                   sizeof(response));
                    GGML_ASSERT(status);
                    static const bool dbg_alloc = getenv("RPC_DBG_ALLOC") != nullptr;
                    if (dbg_alloc && id == 0) {
                        GGML_LOG_INFO("[ALLOC] init_tensor name=%s op=%d size=%zu remote_ptr=0x%llx\n", tensor->name,
                                      (int) tensor->op, size, (unsigned long long) response.remote_ptr);
                    }
                    if (response.remote_ptr != 0) {
                        auto * nctx           = new ggml_backend_rpc_buffer_context{ socks[id], nullptr, response.remote_ptr };
                        extra->buffer_ctx[id] = nctx;
                        extra->rows[id]       = { 0, tensor->ne[0] };
                        if (g_persist_buffers) {
                            std::lock_guard<std::mutex> lock(g_persist_mtx);
                            g_persist_buf[std::make_pair((const void *) tensor, id)] = { nctx, size };
                            g_persist_ctxs.insert(nctx);
                        }
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
                    // home device by ctx-object identity, not remote_ptr value (alias-safe)
                    const void * home_ctx = tensor->buffer ? tensor->buffer->context : nullptr;
                    if (home_ctx != static_cast<const void *>(extra->buffer_ctx[id])) {
                        request.tensor.buffer = extra->buffer_ctx[id]->remote_ptr;
                        // GGML_LOG_INFO("init\n");
                        request.tensor.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                                                  reinterpret_cast<ggml_backend_rpc_buffer_context *>(
                                                      extra->buffer_ctx[id]))) +
                                              extra->data_off;  // (activation pool) sub-allocated offset
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

// Weight-cache key kinds. Folded into the content hash so a WHOLE (non-split / pipeline)
// tensor, a SPLIT (TP) slice, and different split parts never share a cache entry even if
// their bytes coincide (e.g. constant / zero tensors). Content stays in the key, so a model
// swap (different bytes) still misses -> correct across models.
enum rpc_wkind : uint8_t { RPC_WKIND_WHOLE = 0, RPC_WKIND_SPLIT = 1 };

static uint64_t rpc_weight_key(const void * data, size_t size, uint8_t kind, uint8_t device_id, int8_t split_dim) {
    uint64_t      h       = rpc_fnv1a(data, size);
    const uint8_t desc[3] = { kind, device_id, (uint8_t) split_dim };  // fold the descriptor in
    for (size_t i = 0; i < sizeof(desc); ++i) {
        h ^= desc[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// ---- batched warm load (client) -------------------------------------------
// A pipeline (-sm layer) load streamed ~1 weight/tensor through set_tensor, each a synchronous
// LOAD_CACHED round-trip -> ~1 s/tensor, dominated by WiFi round-trips, not the ~25 s of local
// SD reads. Instead DEFER each weight into a per-server batch and flush it as ONE
// RPC_CMD_BATCH_LOAD_CACHED: the server loads every hit from its LOCAL disk (no per-tensor
// round-trip); the client uploads only the misses. We keep a COPY of each tensor's bytes (a
// cheap RAM memcpy of the loader's already-paged data): the loader unmaps its mmap at the end
// of load_all_data, so a raw src pointer would dangle by graph_compute (miss uploads would
// SIGSEGV). To bound memory we flush a server's batch once it reaches ~64 MB (during load);
// the residual is flushed at the first graph_compute, before any compute reads the weights.
struct rpc_pending_load {
    rpc_tensor           rt;
    uint64_t             key;
    std::vector<uint8_t> data;  // owned copy (the loader's mmap is released before we flush)
};
static const size_t                                                  RPC_BATCH_FLUSH_BYTES = 64 * 1024 * 1024;
static std::mutex                                                     g_pending_mtx;
static std::unordered_map<std::string, std::vector<rpc_pending_load>> g_pending_loads;
static std::unordered_map<std::string, size_t>                       g_pending_bytes;
static std::atomic<bool>                                             g_have_pending{ false };

// Flush one server's batch: query all keys in one message (server loads hits from local disk),
// then upload the misses from our owned copies. `list` is consumed.
static void rpc_flush_endpoint(const std::string & endpoint, std::vector<rpc_pending_load> & list) {
    const uint32_t n = (uint32_t) list.size();
    if (n == 0) {
        return;
    }
    auto sock = get_socket(endpoint);
    if (sock) {
        const size_t         entry = sizeof(rpc_tensor) + sizeof(uint64_t);
        std::vector<uint8_t> req(sizeof(uint32_t) + (size_t) n * entry);
        memcpy(req.data(), &n, sizeof(n));
        size_t off = sizeof(uint32_t);
        for (const auto & e : list) {
            memcpy(req.data() + off, &e.rt, sizeof(rpc_tensor));
            off += sizeof(rpc_tensor);
            memcpy(req.data() + off, &e.key, sizeof(uint64_t));
            off += sizeof(uint64_t);
        }
        std::vector<uint8_t> hits(n, 0);
        bool                 ok = send_rpc_cmd(sock, RPC_CMD_BATCH_LOAD_CACHED, req.data(), req.size(), hits.data(), hits.size());
        int                  n_hit = 0, n_miss = 0;
        for (uint32_t i = 0; i < n; ++i) {
            rpc_pending_load & e = list[i];
            if (ok && hits[i]) {
                n_hit++;
                continue;  // server loaded it from its local cache
            }
            n_miss++;
            // MISS: upload + persist (SET_TENSOR_CACHE): | rpc_tensor | offset(8)=0 | key(8) | data |
            std::vector<uint8_t> in(sizeof(rpc_tensor) + 2 * sizeof(uint64_t) + e.data.size());
            uint64_t             offset0 = 0;
            memcpy(in.data(), &e.rt, sizeof(rpc_tensor));
            memcpy(in.data() + sizeof(rpc_tensor), &offset0, sizeof(offset0));
            memcpy(in.data() + sizeof(rpc_tensor) + sizeof(offset0), &e.key, sizeof(uint64_t));
            memcpy(in.data() + sizeof(rpc_tensor) + 2 * sizeof(uint64_t), e.data.data(), e.data.size());
            send_rpc_cmd(sock, RPC_CMD_SET_TENSOR_CACHE, in.data(), in.size(), nullptr, 0);
        }
        static const bool dbg = getenv("RPC_DBG_WCACHE") != nullptr;
        if (dbg) {
            GGML_LOG_INFO("[wcache] BATCH %s: %d hit, %d miss (of %u)\n", endpoint.c_str(), n_hit, n_miss, n);
        }
    }
    list.clear();
}

static void rpc_queue_cached_load(const std::string & endpoint, const rpc_tensor & rt, uint64_t key,
                                  const void * src, size_t size) {
    std::vector<rpc_pending_load> to_flush;  // swapped out under the lock, flushed after releasing it
    std::string                   flush_ep;
    {
        std::lock_guard<std::mutex> lk(g_pending_mtx);
        auto &                      list = g_pending_loads[endpoint];
        list.push_back({ rt, key, std::vector<uint8_t>((const uint8_t *) src, (const uint8_t *) src + size) });
        g_pending_bytes[endpoint] += size;
        g_have_pending.store(true);
        if (g_pending_bytes[endpoint] >= RPC_BATCH_FLUSH_BYTES) {
            to_flush.swap(list);
            g_pending_bytes[endpoint] = 0;
            flush_ep                  = endpoint;
        }
    }
    if (!to_flush.empty()) {
        rpc_flush_endpoint(flush_ep, to_flush);
    }
}

// Flush every remaining server batch (the residual under the byte threshold). Called at the top
// of graph_compute so all weights are in place before any compute.
static void rpc_flush_pending_loads() {
    if (!g_have_pending.exchange(false)) {
        return;
    }
    std::unordered_map<std::string, std::vector<rpc_pending_load>> batches;
    {
        std::lock_guard<std::mutex> lk(g_pending_mtx);
        batches.swap(g_pending_loads);
        g_pending_bytes.clear();
    }
    for (auto & kv : batches) {
        rpc_flush_endpoint(kv.first, kv.second);
    }
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    // GGML_LOG_INFO("[%s] setting tensor %s, offset=%zu, size=%zu\n", __func__, tensor->name, offset, size);
    // GGML_LOG_INFO("ne0 = %ld ne1 = %ld nb0 = %ld nb1 =%ld nb2 = %ld\n",tensor->ne[0],tensor->ne[1],tensor->nb[0],tensor->nb[1],tensor->nb[2]);
    ggml_backend_rpc_buffer_type_context * buft_ctx   = (ggml_backend_rpc_buffer_type_context *) buffer->buft->context;
    ggml_backend_rpc_buffer_context *      ctx        = (ggml_backend_rpc_buffer_context *) buffer->context;
    rpc_tensor rpc_tensor1 = serialize_tensor(tensor);

    // Weight cache on the NON-SPLIT (pipeline / -sm layer) load path. The split path caches in
    // cache_or_upload, but pipeline weights came through here UNcached -> every -sm layer load
    // re-uploaded the whole model over WiFi (~14 min). Only WEIGHTS buffers, only whole-tensor
    // loads (offset 0): ask the server by content key whether it already has this tensor on
    // disk; HIT -> skip the upload, MISS -> upload via SET_TENSOR_CACHE (persist). RPC_WKIND_WHOLE
    // keeps these keys in a separate namespace from TP split slices. Off: RPC_NO_WEIGHT_CACHE.
    // !split: only the pipeline path needs this. In a TP (-sm row) run the split slices are
    // already cached in cache_or_upload, and the replicated non-split tensors must stay on the
    // plain SET_TENSOR path (routing them through SET_TENSOR_CACHE corrupts -sm row output).
    static const bool weight_cache = rpc_opt_enabled() && (getenv("RPC_NO_WEIGHT_CACHE") == nullptr);
    if (weight_cache && !split && offset == 0 &&
        ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        // Defer this weight's cache load into the per-server batch; flushed as ONE
        // RPC_CMD_BATCH_LOAD_CACHED at graph_compute so the server loads all hits from its local
        // disk in one shot (no per-tensor round-trip). Misses are uploaded there via the mmap src.
        const uint64_t key = rpc_weight_key(data, size, RPC_WKIND_WHOLE, 0, -1);
        rpc_queue_cached_load(buft_ctx->endpoint, rpc_tensor1, key, data, size);
        return;
    }

    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    size_t               input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
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
        ggml_tensor_extra_rpc * extra         = (ggml_tensor_extra_rpc *) tensor->extra;
        const int               n_dev         = ggml_backend_rpc_get_device_count();
        // Upload the replicated copy to every other device concurrently. Pre-fetch
        // and HOLD the sockets on the main thread first (the worker threads must
        // not race get_socket / let the weak_ptr-cached sockets churn -- see the
        // split upload). RPC_SERIAL_UPLOAD=1 reverts to one-at-a-time.
        static const bool       serial_upload = (!rpc_opt_enabled() || getenv("RPC_SERIAL_UPLOAD") != nullptr);
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
                rpc_tensor2.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                                       reinterpret_cast<ggml_backend_rpc_buffer_context *>(extra->buffer_ctx[id]))) +
                                   extra->data_off;  // (activation pool) sub-allocated offset
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
        // Cross-server copy == the pipeline stage->stage handoff. Instead of relaying the
        // bytes src -> client(coordinator) -> dst, ask the SRC server to push them STRAIGHT
        // to the DST peer (over the peer link create_peer_connection already dialed). The
        // activation then crosses the shared channel once, not twice, and never touches the
        // coordinator's link. Falls back to the generic get/set relay (return false) when opt
        // is off, the peer link is missing, or the push fails -- so it stays correct always.
        static const bool direct_handoff = rpc_opt_enabled() && getenv("RPC_NO_DIRECT_HANDOFF") == nullptr;
        if (split || !direct_handoff) {
            return false;  // TP (split) keeps the original relay fallback; only pipeline pushes direct
        }
        ggml_backend_rpc_buffer_type_context * dst_buft =
            (ggml_backend_rpc_buffer_type_context *) dst_buffer->buft->context;
        rpc_msg_send_to_peer_req req;
        req.src = serialize_tensor(src);
        req.dst = serialize_tensor(dst);
        snprintf(req.dst_endpoint, sizeof(req.dst_endpoint), "%s", dst_buft->endpoint.c_str());
        rpc_msg_send_to_peer_rsp resp;
        resp.result = 0;
        bool status = send_rpc_cmd(src_ctx->sock, RPC_CMD_SEND_TO_PEER, &req, sizeof(req), &resp, sizeof(resp));
        if (!status || !resp.result) {
            return false;  // fall back to the client relay
        }
        return true;
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
                                       reinterpret_cast<ggml_backend_rpc_buffer_context *>(src_extra->buffer_ctx[id]))) +
                                   src_extra->data_off;  // (activation pool) sub-allocated offset
            }

            if (dst_extra->buffer_ctx[id] == nullptr) {
                GGML_LOG_INFO("[%s] buffer context for device %d is null\n", __func__, id);
            } else {
                request.dst.buffer = dst_extra->buffer_ctx[id]->remote_ptr;
                // GGML_LOG_INFO("cpy\n");
                request.dst.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                                       reinterpret_cast<ggml_backend_rpc_buffer_context *>(dst_extra->buffer_ctx[id]))) +
                                   dst_extra->data_off;  // (activation pool) sub-allocated offset
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
// atomic: the threaded split-buffer alloc workers raise this concurrently (rpc_note_split_block
// from rpc_get_{row,col}_split), and the graph-send build threads read it -- a plain int64
// read-modify-write races and can LOSE a max update -> too-small split block -> wrong split
// boundaries -> garbage/OOB. (Found by ThreadSanitizer.)
static std::atomic<int64_t> g_rpc_split_block{ 1 };

static inline void rpc_note_split_block(const ggml_tensor * tensor) {
    const int64_t blk = ggml_blck_size(tensor->type);
    int64_t       cur = g_rpc_split_block.load(std::memory_order_relaxed);
    while (blk > cur && !g_rpc_split_block.compare_exchange_weak(cur, blk, std::memory_order_relaxed)) {
        // cur reloaded by compare_exchange_weak on failure; retry until blk <= cur or we win
    }
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
        static const bool serial_alloc = (!rpc_opt_enabled() || getenv("RPC_SERIAL_UPLOAD") != nullptr);
        const int         n_dev        = ggml_backend_rpc_get_device_count();
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
    static const bool serial_upload = (!rpc_opt_enabled() || getenv("RPC_SERIAL_UPLOAD") != nullptr);
    const int         n_dev         = ggml_backend_rpc_get_device_count();
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
    static const bool weight_cache = rpc_opt_enabled() && (getenv("RPC_NO_WEIGHT_CACHE") == nullptr);

    // Given a contiguous slice, either skip it (server cache hit) or upload it.
    auto cache_or_upload = [&](int id, const rpc_tensor & rt, const uint8_t * slice, size_t slice_size) {
        if (weight_cache) {
            const uint64_t          hash = rpc_weight_key(slice, slice_size, RPC_WKIND_SPLIT,
                                                          (uint8_t) id, (int8_t) (extra ? extra->split_dim : -1));
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
    static const bool serial_download = (!rpc_opt_enabled() || getenv("RPC_SERIAL_UPLOAD") != nullptr);
    const int         n_dev           = ggml_backend_rpc_get_device_count();
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
                        // Head-split the transposed V cache. The full cache_v_l is channel-major
                        // ([n_embd_v channels] x [n_ctx positions], each channel's positions
                        // contiguous), so each device's flat 1/N slice is naturally its head's
                        // channels x the FULL n_ctx. Keep the full per-channel stride nb[1]; do NOT
                        // scale it by the head fraction. The old `nb[1] *= split_part` reinterpreted
                        // the slice as [all channels] x [n_ctx/N positions], so for n_kv > n_ctx/N the
                        // per-position read/write overflowed into the next channel -> garbage at long
                        // sequences (threshold n_ctx/N; worse as N grows). Applied to BOTH this read
                        // view (v-N) and the write view (v_cache_view) so the stored layout agrees.
                        rpc_t.nb[2] = rpc_t.nb[1] * rpc_t.ne[1];
                        rpc_t.nb[3] = rpc_t.nb[2] * rpc_t.ne[2];
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
                        const int64_t sblk = g_rpc_split_block.load(std::memory_order_relaxed);
                        if (devs > 0 && full / devs >= sblk) {
                            const int64_t align = std::max(rpc_get_col_rounding(tensor_splits), sblk);
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
                        // Head-split the transposed V cache. The full cache_v_l is channel-major
                        // ([n_embd_v channels] x [n_ctx positions], each channel's positions
                        // contiguous), so each device's flat 1/N slice is naturally its head's
                        // channels x the FULL n_ctx. Keep the full per-channel stride nb[1]; do NOT
                        // scale it by the head fraction. The old `nb[1] *= split_part` reinterpreted
                        // the slice as [all channels] x [n_ctx/N positions], so for n_kv > n_ctx/N the
                        // per-position read/write overflowed into the next channel -> garbage at long
                        // sequences (threshold n_ctx/N; worse as N grows). Applied to BOTH this read
                        // view (v-N) and the write view (v_cache_view) so the stored layout agrees.
                        rpc_t.nb[2] = rpc_t.nb[1] * rpc_t.ne[1];
                        rpc_t.nb[3] = rpc_t.nb[2] * rpc_t.ne[2];
                    }
                    return 0;
                }
            }
            break;

        case GGML_OP_PERMUTE:
            {
                // ggml_permute(a, ax0..ax3): a's dim i goes to result dim op_params[i]. Map by the
                // AXES (op_params), not by matching ne VALUES -- ne-matching is ambiguous when two
                // dims share a size (e.g. head_dim == n_tokens at -ub == head_dim), and the buggy
                // last-match-wins gave q-0 nb[0]=row-stride instead of type_size -> mul_mat aborts
                // on multi-ubatch prefill. Axes are unambiguous and match for every n_tokens.
                rpc_tensor      src_tensor = visited[tensor->src[0]];
                const int32_t * axes       = (const int32_t *) tensor->op_params;
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    rpc_t.ne[axes[i]] = src_tensor.ne[i];
                    rpc_t.nb[axes[i]] = src_tensor.nb[i];
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
                    // home device by ctx-object identity, not remote_ptr value (see add_tensor_part main branch)
                    const void * src_home_ctx = src->buffer ? src->buffer->context : nullptr;
                    if (src_home_ctx != static_cast<const void *>(src_extra->buffer_ctx[id])) {
                        src_tensor.buffer = src_extra->buffer_ctx[id]->remote_ptr;
                        src_tensor.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                                              reinterpret_cast<ggml_backend_rpc_buffer_context *>(
                                                  src_extra->buffer_ctx[id]))) +
                                          src_extra->data_off;  // (activation pool) sub-allocated offset
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
                        // home device by ctx-object identity, not remote_ptr value (alias-safe)
                        const void * svs_home_ctx = src->view_src->buffer ? src->view_src->buffer->context : nullptr;
                        if (svs_home_ctx != static_cast<const void *>(src_view_src_extra->buffer_ctx[id])) {
                            src_view_tensor.buffer = src_view_src_extra->buffer_ctx[id]->remote_ptr;
                            src_view_tensor.data = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                                                       reinterpret_cast<ggml_backend_rpc_buffer_context *>(
                                                           src_view_src_extra->buffer_ctx[id]))) +
                                                   src_view_src_extra->data_off;  // (activation pool)
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
        // Identify the "home" device (the buffer serialize_tensor read) by buffer-context OBJECT
        // IDENTITY, not remote_ptr VALUE. The servers are identical Pis whose allocators hand back
        // identical addresses, so remote_ptr collides across devices; a value compare misclassifies
        // a non-home device as home -> it skips the remap and ships a pointer still anchored to the
        // OTHER device's base -> the intermittent warmup deserialize OOB (data << buffer_start).
        // Object identity is unambiguous even under aliasing. For a split buffer tensor->buffer->context
        // is a different type/object than any buffer_ctx[id], so nothing matches and every device takes
        // the full remap below -- which is provably the same pointer the home branch would compute.
        const void * home_ctx = tensor->buffer ? tensor->buffer->context : nullptr;
        if (home_ctx != static_cast<const void *>(tensor_extra->buffer_ctx[id])) {
            rpc_t.buffer = tensor_extra->buffer_ctx[id]->remote_ptr;
            rpc_t.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                               reinterpret_cast<ggml_backend_rpc_buffer_context *>(tensor_extra->buffer_ctx[id]))) +
                         tensor_extra->data_off;  // (activation pool) sub-allocated offset
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
        static const bool dbg_getbase = getenv("RPC_DBG_GETBASE") != nullptr;
        if (dbg_getbase && (tensor->name[0] == 'k' || tensor->name[0] == 'v')) {
            ggml_backend_rpc_buffer_context * bc =
                reinterpret_cast<ggml_backend_rpc_buffer_context *>(tensor_extra->buffer_ctx[id]);
            GGML_LOG_INFO("[KVVIEW] id=%d name=%s op=%d ctx=%p remote_ptr=0x%llx base=%p data=0x%llx voff=%llu\n", id,
                          tensor->name, (int) tensor->op, (void *) bc, (unsigned long long) bc->remote_ptr,
                          bc->base_ptr.load(std::memory_order_relaxed), (unsigned long long) rpc_t.data,
                          (unsigned long long) rpc_t.view_offs);
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
            // home device by ctx-object identity, not remote_ptr value (alias-safe)
            const void * vs_home_ctx = tensor->view_src->buffer ? tensor->view_src->buffer->context : nullptr;
            if (vs_home_ctx != static_cast<const void *>(view_src_extra->buffer_ctx[id])) {
                view_tensor.buffer = view_src_extra->buffer_ctx[id]->remote_ptr;
                view_tensor.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                                         reinterpret_cast<ggml_backend_rpc_buffer_context *>(
                                             view_src_extra->buffer_ctx[id]))) +
                                     view_src_extra->data_off;  // (activation pool) sub-allocated offset
            }
        }
        visited[tensor->view_src] = view_tensor;
        tensors.push_back(view_tensor);
    }
}

// Build the per-device rpc_tensor array for one graph segment [low..high], exactly
// as the tensor-parallel graph-send does. Factored out so the full graph-send (MISS)
// and the diff-cache patch (HIT) build byte-identical tensor arrays -- the patch
// path diffs this token's array against the last one to find the changed view_offs.
static void build_segment_tensors(const ggml_cgraph * cgraph, uint32_t low, uint32_t high, int id,
                                  std::vector<rpc_tensor> & tensors) {
    std::map<ggml_tensor *, rpc_tensor> visited;
    for (uint32_t count = low; count <= high; count++) {
        ggml_tensor * node = cgraph->nodes[count];
        if (node->src[0] != nullptr && node->src[0]->buffer != nullptr &&
            ggml_backend_buft_is_rpc_split(node->src[0]->buffer->buft) &&
            (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) {
            // This node's src0 (weight) lives in a per-device SPLIT buffer, so it must be
            // split-serialized to match the half-size remote buffer -- regardless of whether the
            // node itself is empty. On the LAST ubatch of a multi-ubatch prefill, n_outputs pruning
            // makes the final layer's matmul empty (0 tokens); the old !ggml_is_empty(node) guard
            // dropped it to the non-split path, which shipped the FULL weight into the half buffer ->
            // deserialize OOB (GGML_ASSERT data+size <= buffer_end). An empty node still serializes
            // fine through the split path (change_ne_and_nb carries the 0 dim through) and the server
            // skips its compute, so splitting src0 is always correct.
            ggml_tensor_extra_rpc * node_extra = (ggml_tensor_extra_rpc *) node->src[0]->extra;
            add_tensor_part(node, tensors, visited, node_extra->split_dim, id);
        } else {
            // This node is not eligible for RPC splitting
            add_tensor_part(node, tensors, visited, -1, id);
        }
    }
}

// Build the client's rpc_tensor array for a graph (in the order the server's by_idx mirrors). The
// PP diff cache diffs this array token-to-token; serialize_graph then ships it (MISS) or the diff.
static std::vector<rpc_tensor> build_graph_tensors(const ggml_cgraph * cgraph) {
    std::vector<rpc_tensor>           tensors;
    std::unordered_set<ggml_tensor *> visited;
    for (uint32_t i = 0; i < (uint32_t) cgraph->n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    return tensors;
}

// serialization format: | n_nodes(4) | nodes(n_nodes*8) | n_tensors(4) | tensors(n_tensors*rpc_tensor) |
static void serialize_graph_from_tensors(const ggml_cgraph * cgraph, const std::vector<rpc_tensor> & tensors,
                                         std::vector<uint8_t> & output) {
    uint32_t n_nodes     = cgraph->n_nodes;
    uint32_t n_tensors   = tensors.size();
    size_t   output_size = sizeof(uint32_t) + ((size_t) n_nodes * sizeof(uint64_t)) + sizeof(uint32_t) +
                         ((size_t) n_tensors * sizeof(rpc_tensor));
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, sizeof(n_nodes));
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + sizeof(n_nodes) + (i * sizeof(uint64_t)), &cgraph->nodes[i], sizeof(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.data() + sizeof(n_nodes) + ((size_t) n_nodes * sizeof(uint64_t)));
    *out_ntensors           = n_tensors;
    rpc_tensor * out_tensors =
        (rpc_tensor *) (output.data() + sizeof(n_nodes) + ((size_t) n_nodes * sizeof(uint64_t)) + sizeof(uint32_t));
    memcpy(out_tensors, tensors.data(), (size_t) n_tensors * sizeof(rpc_tensor));
}

static void serialize_graph(const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    serialize_graph_from_tensors(cgraph, build_graph_tensors(cgraph), output);
}

// (PP diff cache) topology hash over a graph's nodes -- excludes the per-token-varying view_offs/
// data/op_params, so a structurally-identical decode graph keys to the same stored graph. ne/nb ARE
// hashed, so a KV-padding-boundary shape change re-MISSes (re-ships). Mirrors the split-path hash.
static uint64_t rpc_graph_topo_hash(const ggml_cgraph * cgraph) {
    uint64_t h   = 1469598103934665603ULL;
    auto     mix = [&](const void * p, size_t n) {
        const uint8_t * b = (const uint8_t *) p;
        for (size_t i = 0; i < n; i++) {
            h ^= b[i];
            h *= 1099511628211ULL;
        }
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * nd = cgraph->nodes[i];
        mix(&nd->op, sizeof(nd->op));
        mix(&nd->type, sizeof(nd->type));
        mix(nd->ne, sizeof(nd->ne));
        mix(nd->nb, sizeof(nd->nb));
        mix(nd->name, sizeof(nd->name));
    }
    return h;
}

// did an rpc_tensor's server-applied value-fields change vs last token? (NOT the pointer wiring)
static bool rpc_tensor_changed(const rpc_tensor & a, const rpc_tensor & b) {
    return a.data != b.data || a.view_offs != b.view_offs || a.flags != b.flags ||
           memcmp(a.ne, b.ne, sizeof(a.ne)) != 0 || memcmp(a.nb, b.nb, sizeof(a.nb)) != 0 ||
           memcmp(a.op_params, b.op_params, sizeof(a.op_params)) != 0;
}

// (PP diff cache) per-server client state: topology -> graph_number, and each graph_number's last-
// sent baseline (to diff against). Keyed by endpoint because layer-distributed PP has many servers.
struct pp_diff_state {
    std::unordered_map<uint64_t, uint8_t>                pp_cache;      // topology hash -> graph_number
    std::unordered_map<uint8_t, std::vector<rpc_tensor>> last_sent;     // graph_number -> baseline array
    std::unordered_map<uint8_t, std::vector<rpc_tensor>> last_pred;     // (prefetch) graph_number -> predicted next-token array
    uint8_t                                              next_gnum = 0;
};

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
        request.tensor        = serialize_tensor(tensor);
        // home device by ctx-object identity, not remote_ptr value (alias-safe)
        const void * home_ctx = tensor->buffer ? tensor->buffer->context : nullptr;
        if (home_ctx != static_cast<const void *>(extra->buffer_ctx[id])) {
            request.tensor.buffer = extra->buffer_ctx[id]->remote_ptr;
            request.tensor.data   = reinterpret_cast<uint64_t>(ggml_backend_rpc_buffer_context_get_base(
                                      reinterpret_cast<ggml_backend_rpc_buffer_context *>(extra->buffer_ctx[id]))) +
                                  extra->data_off;  // (activation pool) sub-allocated offset
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
            tensor_cpy->data   = (void *) ((uint64_t) ggml_backend_rpc_buffer_context_get_base(extra->buffer_ctx[id]) +
                                           extra->data_off);  // (activation pool) sub-allocated offset
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
// RPC_DBG_TIMING: per-token split of the client's wall-time -- GRAPH_COMPUTE
// round-trips (graph transfer+store) vs DO_COMPUTATION round-trips (servers'
// execute + peer-to-peer all-reduce). Cumulative; logged once per token.
static std::atomic<long long> g_graph_send_ns{ 0 };
static std::atomic<long long> g_do_comp_ns{ 0 };
static std::atomic<int>       g_compute_tokens{ 0 };
// per-token client-side phase WALL breakdown (RPC_DBG_TIMING): where the decode token actually goes
static std::atomic<long long> g_build_ns{ 0 };  // HIT build phase wall (build_segment_tensors + diff + prefetch-verify)
static std::atomic<long long> g_send_ns{ 0 };   // graph-send phase wall (PATCH/ADVANCE; ~0 with oneway)
static std::atomic<long long> g_compute_ns{ 0 };  // DO_COMPUTATION phase wall (all-reduce + result-gather)
static std::atomic<long long> g_gather_ns{ 0 };   // result-gather (add_data_to_data) sum across devices
static std::atomic<long long> g_bst_ns{ 0 };      // build_segment_tensors only (summed over device threads; wall ~ /N)

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // GGML_LOG_INFO("graph compute for cgraph %x\n", (uint64_t) cgraph);
    // Flush any deferred weight loads (batched warm load) before the first compute -- one
    // RPC_CMD_BATCH_LOAD_CACHED per server loads all its cache hits from local disk, so every
    // weight is in place before the graph runs. No-op once flushed (g_have_pending is cleared).
    rpc_flush_pending_loads();
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
        // RPC_GRAPH_CACHE: the DIFF cache. The graph-send (~20% of decode) re-ships the
        // whole ~1000-tensor structure every token even though only the KV-cache write
        // views change -- their view_offs (and derived data pointer) advance one slot per
        // token; everything else (op/type/shape/strides/wiring) is identical across decode
        // tokens within a KV padding window. So: ship the structure ONCE (MISS), keyed by a
        // topology hash that EXCLUDES the per-token state; on a later same-topology token
        // (HIT) re-send only the changed tensors as RPC_CMD_PATCH_VIEWS and let the server
        // patch its stored graph in place -- no re-deserialize. When the KV length crosses a
        // padding boundary the shapes change -> new topology -> a fresh MISS re-stores it.
        // (Earlier a whole-graph skip excluding view_offs gave false hits that re-ran a
        // stale-KV-position graph -> garbage; the diff cache fixes that by patching the
        // positions every token.) RPC_DBG_GHASH logs the view_offs-inclusive probe hash;
        // RPC_DBG_DIFFCACHE logs hit/miss + patch counts. All default off.
        static const bool dbg_ghash      = getenv("RPC_DBG_GHASH") != nullptr;
        // Under the master gate the diff cache is ON by default (optimized); RPC_NO_OPT
        // turns it off (baseline ships the full graph every token).
        static const bool use_diff_cache = rpc_opt_enabled();
        // (#3) prefetch: predict the next-token patch + ship a 1-byte GRAPH_ADVANCE when it holds.
        // DEFAULT ON with the diff cache; opt out with RPC_NO_PREFETCH. RPC_PREFETCH_DBG logs the
        // prediction match rate (needs RPC_PERSIST_BUFFERS for the prediction to actually hold).
        static const bool prefetch       = use_diff_cache && getenv("RPC_NO_PREFETCH") == nullptr;
        static const bool prefetch_dbg   = getenv("RPC_PREFETCH_DBG") != nullptr;
        // (#3b) skip the per-token client REBUILD. Once the predictor has held for a few tokens the
        // verify-rebuild is redundant: advance the client's prev/pred arithmetically (cheap, no
        // cgraph walk) and send ADVANCE blindly -- the server advances its stored graph by the same
        // cached stride a verified ADVANCE would have. Re-verify (full rebuild) every
        // RPC_SKIP_VERIFY_PERIOD tokens (0 = never until a MISS) and always at the deterministic
        // MISS boundary. Requires RPC_PREFETCH. Biggest decode lever: the rebuild is ~3.4 s/tok of
        // coordinator CPU, far above the all-reduce.
        static const bool skip_build     = prefetch && getenv("RPC_PREFETCH_SKIP_BUILD") != nullptr;
        static const int  skip_verify_period = []() {
            const char * e = getenv("RPC_SKIP_VERIFY_PERIOD");
            return e ? atoi(e) : 32;
        }();
        static const int  skip_warmup    = 3;  // consecutive fully-verified ADVANCE tokens before skipping
        static const bool dbg_diffcache  = getenv("RPC_DBG_DIFFCACHE") != nullptr;
        static std::unordered_map<uint64_t, uint8_t>
            graph_cache;  // topology hash (excl. view_offs) -> server graph_number
        // server graph_number -> [segment][device] -> rpc_tensor array we last shipped, so a
        // HIT can diff this token's array against it and patch only what changed.
        static std::unordered_map<uint8_t, std::vector<std::vector<std::vector<rpc_tensor>>>> last_sent;
        // (#3 prefetch) the patch we PREDICT for the NEXT token, computed this token as
        // current + (current - prev) per advancing field (Phase-0: constant per-tensor stride).
        // Carried across tokens so the next token can (a) verify the prediction and (b) skip the
        // graph-send when it holds. Sized like last_sent; only maintained when RPC_PREFETCH is set.
        static std::unordered_map<uint8_t, std::vector<std::vector<std::vector<rpc_tensor>>>> last_pred;
        // (#3b skip-build) state: consecutive fully-verified ADVANCE tokens, and tokens since the
        // last full rebuild (re-verify). Reset on any MISS (boundary), where the graph is rebuilt.
        static int skip_consec_ok    = 0;
        static int skip_since_verify = 0;

        uint64_t struct_hash = 0;
        if (dbg_ghash) {
            struct_hash = 1469598103934665603ULL;  // FNV-1a over op/type/shape/name (no pointers)
            auto mix    = [&](const void * p, size_t n) {
                const uint8_t * b = (const uint8_t *) p;
                for (size_t i = 0; i < n; i++) {
                    struct_hash ^= b[i];
                    struct_hash *= 1099511628211ULL;
                }
            };
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * nd = cgraph->nodes[i];
                mix(&nd->op, sizeof(nd->op));
                mix(&nd->type, sizeof(nd->type));
                mix(nd->ne, sizeof(nd->ne));
                mix(nd->nb, sizeof(nd->nb));                 // strides
                mix(&nd->view_offs, sizeof(nd->view_offs));  // KV-cache write position advances per token
                mix(nd->name, sizeof(nd->name));
            }
        }
        if (dbg_ghash) {
            static uint64_t prev_h = 0;
            GGML_LOG_INFO("[rpc-ghash] cgraph=%p n_nodes=%d hash=%016llx %s\n", (void *) cgraph, cgraph->n_nodes,
                          (unsigned long long) struct_hash, prev_h == struct_hash ? "(SAME as prev)" : "(changed)");
            prev_h = struct_hash;
        }

        // RPC_DBG_DIFF: size the diff-cache. Per token, count how many nodes differ from
        // the previous same-topology token, broken down by field. Confirms which fields
        // the patch must ship: measured per decode token only view_offs (~48/40 nodes) and
        // op_params (~24/20) change in the client cgraph; ne/nb/name are stable there (the
        // per-device split-adjusted ne/nb still vary, which is why the patch re-sends them).
        static const bool dbg_diff = getenv("RPC_DBG_DIFF") != nullptr;
        if (dbg_diff) {
            auto fnv = [](const void * p, size_t n) {
                const uint8_t * b = (const uint8_t *) p;
                uint64_t        h = 1469598103934665603ULL;
                for (size_t i = 0; i < n; i++) {
                    h ^= b[i];
                    h *= 1099511628211ULL;
                }
                return h;
            };
            uint64_t topo = 1469598103934665603ULL;  // hash EXCLUDING view_offs (topology only)
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * nd = cgraph->nodes[i];
                for (auto v : { (uint64_t) nd->op, (uint64_t) nd->type, (uint64_t) nd->ne[0], (uint64_t) nd->ne[1] }) {
                    topo ^= v;
                    topo *= 1099511628211ULL;
                }
            }
            static std::unordered_map<uint64_t, std::vector<std::array<uint64_t, 7>>> prevf;  // per-node field snapshot
            auto &                                                                    pf = prevf[topo];
            if ((int) pf.size() == cgraph->n_nodes) {
                int dvo = 0;
                int dne = 0;
                int dnb = 0;
                int dop = 0;
                int dnm = 0;
                for (int i = 0; i < cgraph->n_nodes; i++) {
                    ggml_tensor * nd = cgraph->nodes[i];
                    if ((uint64_t) nd->view_offs != pf[i][0]) {
                        dvo++;
                    }
                    if (fnv(nd->ne, sizeof(nd->ne)) != pf[i][1]) {
                        dne++;
                    }
                    if (fnv(nd->nb, sizeof(nd->nb)) != pf[i][2]) {
                        dnb++;
                    }
                    if (fnv(nd->op_params, sizeof(nd->op_params)) != pf[i][3]) {
                        dop++;
                    }
                    if (fnv(nd->name, sizeof(nd->name)) != pf[i][4]) {
                        dnm++;
                    }
                }
                GGML_LOG_INFO("[rpc-diff] n_nodes=%d view_offs=%d ne=%d nb=%d op_params=%d name=%d\n", cgraph->n_nodes,
                              dvo, dne, dnb, dop, dnm);
            }
            pf.resize(cgraph->n_nodes);
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * nd = cgraph->nodes[i];
                pf[i]            = { (uint64_t) nd->view_offs,
                                     fnv(nd->ne, sizeof(nd->ne)),
                                     fnv(nd->nb, sizeof(nd->nb)),
                                     fnv(nd->op_params, sizeof(nd->op_params)),
                                     fnv(nd->name, sizeof(nd->name)),
                                     0,
                                     0 };
            }
        }

        // Topology hash for the diff cache -- STABLE structure only (op/type/shape/
        // strides/name), deliberately EXCLUDING view_offs, data pointers AND op_params:
        // those three carry the per-token state (KV write position + position offsets) the
        // patch ships. (Measured: per decode token only view_offs (~48 nodes) and op_params
        // (~24 nodes) change; ne/nb/name are stable within a KV padding window.) Same hash
        // across decode tokens => the servers already hold this graph.
        uint64_t topo_hash = 0;
        if (use_diff_cache) {
            topo_hash = 1469598103934665603ULL;
            auto mix  = [&](const void * p, size_t n) {
                const uint8_t * b = (const uint8_t *) p;
                for (size_t i = 0; i < n; i++) {
                    topo_hash ^= b[i];
                    topo_hash *= 1099511628211ULL;
                }
            };
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * nd = cgraph->nodes[i];
                mix(&nd->op, sizeof(nd->op));
                mix(&nd->type, sizeof(nd->type));
                mix(nd->ne, sizeof(nd->ne));
                mix(nd->nb, sizeof(nd->nb));
                mix(nd->name, sizeof(nd->name));
            }
        }

        bool    cache_hit         = false;
        uint8_t this_graph_number = global_graph_number;
        if (use_diff_cache) {
            auto it = graph_cache.find(topo_hash);
            // require last_sent too: we can only patch a graph we still hold the baseline for
            if (it != graph_cache.end() && last_sent.count(it->second)) {
                cache_hit         = true;
                this_graph_number = it->second;
            }
        }

        int device_count = ggml_backend_rpc_get_device_count();

        // Segment boundaries (sync points). Same for the store (MISS) and patch (HIT)
        // paths, so compute once up front.
        std::vector<struct sync_split> sync_splits;
        {
            uint32_t count_nodes_low = 0;
            for (int count_nodes = 0; count_nodes < cgraph->n_nodes; count_nodes++) {
                ggml_tensor * node = cgraph->nodes[count_nodes];
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
                    }
                }
            }
            change_split = sync_splits[sync_splits.size() - 2];
        }

        uint32_t n_segments = (uint32_t) sync_splits.size();

        if (dbg_diffcache) {
            static int hits   = 0;
            static int misses = 0;
            cache_hit ? ++hits : ++misses;
            GGML_LOG_INFO("[rpc-diffcache] %s gnum=%d segs=%u (hits=%d misses=%d)\n", cache_hit ? "HIT " : "MISS",
                          this_graph_number, n_segments, hits, misses);
        }

        // MISS: ship the graph structure to the servers (a cache HIT patches instead).
        if (!cache_hit) {
            // (#3b) boundary: the topology changed -> any skip streak is invalid; re-establish.
            skip_consec_ok    = 0;
            skip_since_verify = 0;
            // BATCHED graph-send: build every segment per device, then send ONE
            // GRAPH_COMPUTE_BATCH per device below (was one RPC per segment*device ~=
            // 176 sequential round-trips/token; the graph-send is round-trip-bound, so
            // this collapses it to one round-trip per device). The server stores each
            // segment exactly as the per-segment path did. Layout per device:
            //   n_segments(4) | per segment: seg_len(4) | seg_data
            int                               batch_dev_count = device_count;
            std::vector<std::vector<uint8_t>> dev_batch(batch_dev_count);
            {
                for (int id = 0; id < batch_dev_count; ++id) {
                    dev_batch[id].resize(sizeof(uint32_t));
                    memcpy(dev_batch[id].data(), &n_segments, sizeof(n_segments));
                }
            }
            // Diff cache: remember each device's per-segment tensor array so the next
            // same-topology token can diff against it and patch only what changed.
            std::vector<std::vector<std::vector<rpc_tensor>>> * sent = nullptr;
            if (use_diff_cache) {
                last_sent[this_graph_number].assign(n_segments, std::vector<std::vector<rpc_tensor>>(batch_dev_count));
                sent = &last_sent[this_graph_number];
                if (prefetch) {  // reset stale predictions on a fresh/re-stored graph
                    last_pred[this_graph_number].assign(n_segments,
                                                        std::vector<std::vector<rpc_tensor>>(batch_dev_count));
                }
            }
            // Master gate (== rpc_opt_enabled()): also controls graph-send BATCHING. opt -> build
            // all segments then one GRAPH_COMPUTE_BATCH/device; baseline -> send each segment
            // immediately as its own RPC_CMD_GRAPH_COMPUTE (~n_segments x n_devices round-trips).
            const bool opt = use_diff_cache;
            // Compute the computation graph for the current split and synchronize results
            for (size_t count_split = 0; count_split < sync_splits.size(); count_split++) {
                // GGML_LOG_INFO("\n------------split-------------\n");
                struct sync_split sync_split      = sync_splits[count_split];
                uint32_t          count_nodes_low = sync_split.nodes_split.first;
                uint32_t          count_nodes     = sync_split.nodes_split.second;
                // GGML_LOG_INFO("low = %d high = %d\n",count_nodes_low,count_nodes);
                // bool              checkend        = sync_split.checkend;

                ggml_tensor * tensor = cgraph->nodes[count_nodes];

                std::vector<std::thread> threads;

                for (int id = 0; id < device_count; ++id) {
                    threads.emplace_back([&, id]() {
                        //we need to compute the next part of the graph
                        std::vector<rpc_tensor> tensors;
                        build_segment_tensors(cgraph, count_nodes_low, count_nodes, id, tensors);

                        std::vector<uint8_t> input;
                        uint32_t             n_nodes = count_nodes - count_nodes_low + 1;
                        if (n_nodes == 0) {
                            return;
                        }

                        //input format: signal(1 byte) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t)) |
                        //              n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) | graph_number (1 byte)
                        uint32_t n_tensors  = tensors.size();
                        int      input_size = sizeof(uint8_t) + sizeof(uint32_t) + (n_nodes * sizeof(uint64_t)) +
                                         sizeof(uint32_t) + (n_tensors * sizeof(rpc_tensor)) + sizeof(uint8_t);
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
                            memcpy(input.data() + sizeof(uint8_t) + sizeof(n_nodes) + (i * sizeof(uint64_t)),
                                   &cgraph->nodes[count_nodes_low + i], sizeof(uint64_t));
                        }

                        // Append number of tensors
                        uint32_t * in_ntensors = (uint32_t *) (input.data() + sizeof(uint8_t) + sizeof(n_nodes) +
                                                               (n_nodes * sizeof(uint64_t)));
                        *in_ntensors           = n_tensors;

                        // Copy tensor metadata
                        rpc_tensor * in_tensors = (rpc_tensor *) (input.data() + sizeof(uint8_t) + sizeof(n_nodes) +
                                                                  (n_nodes * sizeof(uint64_t)) + sizeof(uint32_t));
                        memcpy(in_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));

                        // denote which graph number this is, and info for graph that servers need to know
                        uint8_t * in_graph_number = (uint8_t *) (input.data() + sizeof(uint8_t) + sizeof(n_nodes) +
                                                                 (n_nodes * sizeof(uint64_t)) + sizeof(uint32_t) +
                                                                 (n_tensors * sizeof(rpc_tensor)));
                        *in_graph_number          = this_graph_number;

                        if (opt) {
                            // accumulate this segment into the device's batch (one send below)
                            uint32_t seg_len = (uint32_t) input.size();
                            size_t   base    = dev_batch[id].size();
                            dev_batch[id].resize(base + sizeof(uint32_t) + seg_len);
                            memcpy(dev_batch[id].data() + base, &seg_len, sizeof(uint32_t));
                            memcpy(dev_batch[id].data() + base + sizeof(uint32_t), input.data(), seg_len);
                            // diff cache: keep this segment's array for next token's patch diff
                            if (sent) {
                                (*sent)[count_split][id] = std::move(tensors);
                            }
                        } else {
                            // BASELINE: send this segment now as its own (acked) RPC_CMD_GRAPH_COMPUTE
                            auto * dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
                            auto   sock    = get_socket(dev_ctx->endpoint);
                            rpc_msg_graph_compute_rsp response;
                            auto                      _t_gs = std::chrono::steady_clock::now();
                            bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size(),
                                                       &response, sizeof(response));
                            g_graph_send_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() - _t_gs)
                                                   .count();
                            GGML_ASSERT(status);
                            if (response.result != GGML_STATUS_SUCCESS) {
                                fprintf(stderr, "RPC graph compute failed with status %d\n", response.result);
                            }
                        }
                    });
                }

                // Join all threads (build only -- no network here)
                for (auto & thread : threads) {
                    if (thread.joinable()) {
                        thread.join();
                    }
                }
            }

            // OPTIMIZED: send each device's full batch in ONE round-trip (concurrent across
            // devices). BASELINE already sent each segment above, so nothing to do here.
            if (opt) {
                std::vector<std::thread> send_threads;
                send_threads.reserve(batch_dev_count);
                for (int id = 0; id < batch_dev_count; ++id) {
                    send_threads.emplace_back([&, id]() {
                        auto * dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
                        auto   sock    = get_socket(dev_ctx->endpoint);
                        rpc_msg_graph_compute_rsp response;
                        auto                      _t_gs = std::chrono::steady_clock::now();
                        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE_BATCH, dev_batch[id].data(),
                                                   dev_batch[id].size(), &response, sizeof(response));
                        g_graph_send_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - _t_gs)
                                               .count();
                        GGML_ASSERT(status);
                        if (response.result != GGML_STATUS_SUCCESS) {
                            fprintf(stderr, "RPC graph compute (batch) failed with status %d\n", response.result);
                        }
                    });
                }
                for (auto & t : send_threads) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
            }

            graph_splits[reinterpret_cast<uint64_t>(cgraph)] = global_graph_number;
            if (use_diff_cache) {
                graph_cache[topo_hash] = this_graph_number;  // remember: this topology -> this graph_number
            }
            global_graph_number++;
        } else {
            // HIT: the servers already hold this graph (this_graph_number). Rebuild each
            // device's per-segment tensor array, diff it against what we last shipped, and
            // send only the changed tensors (advancing KV-cache write positions + the
            // split-adjusted ne/nb/op_params) as RPC_CMD_PATCH_VIEWS. The server patches its
            // stored graph in place, skipping the re-deserialize of the whole ~1000-tensor
            // structure. Payload per device: graph_number(1) | n_segments(4) |
            //   per segment: n_patches(4) | rpc_view_patch[n_patches]
            auto &                            sent      = last_sent[this_graph_number];
            auto *                            sent_pred = prefetch ? &last_pred[this_graph_number] : nullptr;
            std::vector<std::vector<uint8_t>> dev_patch(device_count);
            std::vector<char> dev_advance(device_count, prefetch ? 1 : 0);  // (#3) per-device: send ADVANCE not a patch
            std::atomic<int>  total_patches{ 0 };
            std::atomic<int>  pred_ok{ 0 }, pred_total{ 0 };

            // (#3b) decide whether to SKIP the rebuild this token: enabled, warmed up, not due for
            // a periodic re-verify, and every device's prediction is established (advanceable).
            bool do_skip = skip_build && sent_pred && skip_consec_ok >= skip_warmup &&
                           (skip_verify_period <= 0 || skip_since_verify < skip_verify_period);
            for (int id = 0; do_skip && id < device_count; ++id) {
                for (uint32_t s = 0; s < n_segments; ++s) {
                    const auto & prev = sent[s][id];
                    const auto & pred = (*sent_pred)[s][id];
                    if (pred.empty() || pred.size() != prev.size()) { do_skip = false; break; }
                }
            }

            auto                     _t_build = std::chrono::steady_clock::now();
            std::vector<std::thread> build_threads;
            for (int id = 0; id < device_count; ++id) {
                build_threads.emplace_back([&, id]() {
                    // (#3b skip-build) steady state: advance prev/pred arithmetically (no cgraph
                    // walk, no diff) and leave dev_advance[id]=1 so the send ships a 1-byte ADVANCE.
                    // The server advances its stored graph by the same cached stride -> identical
                    // result to a verified ADVANCE, but ~3.4 s/tok of coordinator CPU is skipped.
                    if (do_skip) {
                        for (uint32_t s = 0; s < n_segments; ++s) {
                            std::vector<rpc_tensor> & prev = sent[s][id];
                            std::vector<rpc_tensor> & pred = (*sent_pred)[s][id];
                            for (size_t i = 0; i < prev.size(); i++) {
                                rpc_tensor old_pred = pred[i];
                                rpc_tensor np        = pred[i];  // next pred = pred + (pred - prev)
                                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                                    np.ne[d] = pred[i].ne[d] + (pred[i].ne[d] - prev[i].ne[d]);
                                    np.nb[d] = pred[i].nb[d] + (pred[i].nb[d] - prev[i].nb[d]);
                                }
                                for (size_t j = 0; j < GGML_MAX_OP_PARAMS / sizeof(int32_t); j++) {
                                    np.op_params[j] =
                                        pred[i].op_params[j] + (pred[i].op_params[j] - prev[i].op_params[j]);
                                }
                                np.flags     = pred[i].flags + (pred[i].flags - prev[i].flags);
                                np.data      = pred[i].data + (pred[i].data - prev[i].data);
                                np.view_offs = pred[i].view_offs + (pred[i].view_offs - prev[i].view_offs);
                                prev[i]      = old_pred;  // server now holds the previously-predicted values
                                pred[i]      = np;
                            }
                        }
                        return;
                    }
                    std::vector<uint8_t> & out = dev_patch[id];
                    out.resize(sizeof(uint8_t) + sizeof(uint32_t));
                    out[0] = this_graph_number;
                    memcpy(out.data() + sizeof(uint8_t), &n_segments, sizeof(n_segments));

                    for (size_t s = 0; s < sync_splits.size(); s++) {
                        uint32_t                low  = sync_splits[s].nodes_split.first;
                        uint32_t                high = sync_splits[s].nodes_split.second;
                        std::vector<rpc_tensor> tensors;
                        auto _t_bst = std::chrono::steady_clock::now();
                        build_segment_tensors(cgraph, low, high, id, tensors);
                        g_bst_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - _t_bst).count();

                        std::vector<rpc_tensor> &   prev = sent[s][id];
                        std::vector<rpc_view_patch> patches;
                        // same topology => same count/order; guard defensively against drift
                        size_t                      n = std::min(tensors.size(), prev.size());

                        // A patch carries a tensor whose server-applied value-fields changed (NOT
                        // the pointer wiring id/src/view_src/buffer/name, which always differ).
                        // data_only_out (optional) reports a pure data-pointer change.
                        auto changed_fields = [](const rpc_tensor & a, const rpc_tensor & b, bool * data_only_out) {
                            bool data_changed  = a.data != b.data;
                            bool other_changed = a.view_offs != b.view_offs || a.flags != b.flags ||
                                                 memcmp(a.ne, b.ne, sizeof(a.ne)) != 0 ||
                                                 memcmp(a.nb, b.nb, sizeof(a.nb)) != 0 ||
                                                 memcmp(a.op_params, b.op_params, sizeof(a.op_params)) != 0;
                            if (data_only_out) {
                                *data_only_out = data_changed && !other_changed;
                            }
                            return data_changed || other_changed;
                        };

                        // (#5) data-only trim and (#3 Phase-0) patch classification are both opt-in.
                        // When both are off (the default) take the fast path -- no helper containers,
                        // no per-tensor gate checks, just the plain diff.
                        // (#5) WARNING: EXPERIMENTAL + UNSAFE. Localhost N=4 shows it corrupts decode
                        // ("Athensensens..."): ggml-alloc REUSES buffer slots, so suppressing a
                        // data-only change leaves the server's pointer stale and a later tensor
                        // aliases that slot. Needs global slot-liveness analysis. Do NOT enable.
                        static const bool data_trim = []() {
                            const bool on = rpc_opt_enabled() && getenv("RPC_DIFF_DATA_TRIM") != nullptr;
                            if (on) {
                                GGML_LOG_INFO(
                                    "[RPC_DIFF_DATA_TRIM] WARNING: experimental + UNSAFE "
                                    "(corrupts decode via ggml-alloc slot reuse)\n");
                            }
                            return on;
                        }();
                        static const bool dbg_patch = getenv("RPC_DBG_PATCH") != nullptr;

                        if (!data_trim && !dbg_patch) {
                            // (diag) cgraph pointer stability across tokens: if tensors[i].id (the
                            // ggml_tensor*) matches last token's, an incremental refresh can map
                            // cached[i]->node directly; else it must be position-based.
                            static const bool dbg_idstab = getenv("RPC_DBG_IDSTAB") != nullptr;
                            int               id_same = 0, id_diff = 0;
                            for (size_t i = 0; i < n; i++) {
                                if (dbg_idstab) {
                                    (tensors[i].id == prev[i].id) ? id_same++ : id_diff++;
                                }
                                if (changed_fields(tensors[i], prev[i], nullptr)) {
                                    rpc_view_patch p;
                                    p.idx = (uint32_t) i;
                                    p.t   = tensors[i];
                                    patches.push_back(p);
                                }
                            }
                            if (dbg_idstab && id == 0) {
                                GGML_LOG_INFO("[IDSTAB] gnum=%u seg=%zu n=%zu id_same=%d id_diff=%d\n",
                                              this_graph_number, s, n, id_same, id_diff);
                            }
                        } else {
                            // instrumented / experimental path (helpers built only here)
                            std::unordered_set<uint64_t> viewsrc_ids;  // (#5) keep-set: view-sources
                            uint64_t                     out_id = 0;
                            if (data_trim) {
                                for (const auto & t : tensors) {
                                    if (t.view_src) {
                                        viewsrc_ids.insert(t.view_src);
                                    }
                                }
                                out_id = reinterpret_cast<uint64_t>(cgraph->nodes[high]);
                            }
                            int         c_pos_only = 0, c_data_only = 0, c_mixed = 0;
                            std::string data_sample;
                            for (size_t i = 0; i < n; i++) {
                                const rpc_tensor & a         = tensors[i];
                                const rpc_tensor & b         = prev[i];
                                bool               data_only = false;
                                bool               changed   = changed_fields(a, b, &data_only);
                                if (dbg_patch && changed) {
                                    const bool dc = a.data != b.data;
                                    if (data_only) {
                                        c_data_only++;
                                    } else if (dc) {
                                        c_mixed++;
                                    } else {
                                        c_pos_only++;
                                    }
                                    if (dc && (int) data_sample.size() < 240) {
                                        char buf[96];
                                        snprintf(buf, sizeof(buf), "[%zu]%s 0x%llx->0x%llx ", i, a.name,
                                                 (unsigned long long) b.data, (unsigned long long) a.data);
                                        data_sample += buf;
                                    }
                                }
                                if (changed && data_trim && data_only) {  // (#5) skip plain-intermediate churn
                                    bool keep = a.view_src != 0 || viewsrc_ids.count(a.id) != 0 || a.id == out_id ||
                                                strncmp(a.name, "result", 6) == 0;
                                    if (!keep) {
                                        changed = false;
                                    }
                                }
                                if (changed) {
                                    rpc_view_patch p;
                                    p.idx = (uint32_t) i;
                                    p.t   = a;
                                    patches.push_back(p);
                                }
                            }
                            if (dbg_patch && id == 0) {
                                GGML_LOG_INFO("[PATCH] gnum=%u seg=%zu n=%zu pos_only=%d data_only=%d mixed=%d | %s\n",
                                              this_graph_number, s, patches.size(), c_pos_only, c_data_only, c_mixed,
                                              data_sample.c_str());
                            }
                        }
                        total_patches += (int) patches.size();

                        uint32_t n_patches = (uint32_t) patches.size();
                        size_t   base      = out.size();
                        out.resize(base + sizeof(uint32_t) + n_patches * sizeof(rpc_view_patch));
                        memcpy(out.data() + base, &n_patches, sizeof(n_patches));
                        memcpy(out.data() + base + sizeof(uint32_t), patches.data(),
                               n_patches * sizeof(rpc_view_patch));

                        // (#3 prefetch) verify last token's prediction, then synthesize this
                        // token's prediction for the NEXT token (current + per-field stride).
                        if (prefetch) {
                            std::vector<rpc_tensor> & pred   = (*sent_pred)[s][id];
                            // Verify last token's prediction against the real array. A GRAPH_ADVANCE
                            // is safe for this device only if EVERY tensor's advanced fields match,
                            // so the server's stride-advance reproduces this exact array (any
                            // mismatch -> fall back to a real patch this token).
                            bool                      seg_ok = (pred.size() == tensors.size() && !pred.empty());
                            if (seg_ok || prefetch_dbg) {
                                for (size_t i = 0; i < tensors.size() && i < pred.size(); i++) {
                                    const rpc_tensor & a = tensors[i];
                                    const rpc_tensor & p = pred[i];
                                    bool match = p.data == a.data && p.view_offs == a.view_offs && p.flags == a.flags &&
                                                 memcmp(p.ne, a.ne, sizeof(a.ne)) == 0 &&
                                                 memcmp(p.nb, a.nb, sizeof(a.nb)) == 0 &&
                                                 memcmp(p.op_params, a.op_params, sizeof(a.op_params)) == 0;
                                    if (prefetch_dbg) {  // match-rate over CHANGED tensors only
                                        const rpc_tensor & b  = prev[i];
                                        bool               ch = a.data != b.data || a.view_offs != b.view_offs ||
                                                  a.flags != b.flags || memcmp(a.ne, b.ne, sizeof(a.ne)) != 0 ||
                                                  memcmp(a.nb, b.nb, sizeof(a.nb)) != 0 ||
                                                  memcmp(a.op_params, b.op_params, sizeof(a.op_params)) != 0;
                                        if (ch) {
                                            pred_total++;
                                            if (match) {
                                                pred_ok++;
                                            }
                                        }
                                    }
                                    if (!match) {
                                        seg_ok = false;
                                        if (!prefetch_dbg) {
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!seg_ok) {
                                dev_advance[id] = 0;
                            }  // any mismatch -> real patch for this device
                            // synthesize prediction for token+1: advance every patched field by its stride
                            if (prev.size() == tensors.size()) {
                                pred.resize(tensors.size());
                                for (size_t i = 0; i < tensors.size(); i++) {
                                    const rpc_tensor & a = tensors[i];
                                    const rpc_tensor & b = prev[i];
                                    pred[i]              = a;
                                    for (int d = 0; d < GGML_MAX_DIMS; d++) {
                                        pred[i].ne[d] = a.ne[d] + (a.ne[d] - b.ne[d]);
                                        pred[i].nb[d] = a.nb[d] + (a.nb[d] - b.nb[d]);
                                    }
                                    for (size_t j = 0; j < GGML_MAX_OP_PARAMS / sizeof(int32_t); j++) {
                                        pred[i].op_params[j] = a.op_params[j] + (a.op_params[j] - b.op_params[j]);
                                    }
                                    pred[i].flags     = a.flags + (a.flags - b.flags);
                                    pred[i].data      = a.data + (a.data - b.data);
                                    pred[i].view_offs = a.view_offs + (a.view_offs - b.view_offs);
                                }
                            } else {
                                pred.clear();
                                dev_advance[id] = 0;
                            }
                        }

                        prev = std::move(tensors);  // server now holds these values
                    }
                });
            }
            for (auto & t : build_threads) {
                if (t.joinable()) {
                    t.join();
                }
            }
            g_build_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _t_build)
                    .count();
            // (#3b) update skip state. A skipped token just advances the re-verify clock; a real
            // build counts toward warmup only if EVERY device verified an ADVANCE (prediction held)
            // and resets the re-verify clock (this token IS the re-verify / re-establish).
            if (do_skip) {
                skip_since_verify++;
            } else {
                bool all_adv = prefetch;
                for (int id = 0; id < device_count; ++id) {
                    if (!dev_advance[id]) { all_adv = false; }
                }
                skip_consec_ok    = all_adv ? skip_consec_ok + 1 : 0;
                skip_since_verify = 0;
            }
            if (dbg_diffcache && skip_build) {
                GGML_LOG_INFO("[rpc-skip] gnum=%u %s (consec_ok=%d since_verify=%d)\n", this_graph_number,
                              do_skip ? "SKIP-build (arith advance)" : "build", skip_consec_ok, skip_since_verify);
            }
            if (prefetch_dbg && pred_total.load() > 0) {
                GGML_LOG_INFO("[PREFETCH] gnum=%u predicted %d/%d changed tensors (%.1f%%)\n", this_graph_number,
                              pred_ok.load(), pred_total.load(), 100.0 * pred_ok.load() / pred_total.load());
            }

            // send each device's patch in ONE round-trip (concurrent), timed as graph-send
            auto                     _t_send = std::chrono::steady_clock::now();
            std::vector<std::thread> send_threads;
            for (int id = 0; id < device_count; ++id) {
                send_threads.emplace_back([&, id]() {
                    auto               dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
                    auto               sock    = get_socket(dev_ctx->endpoint);
                    auto               _t_gs   = std::chrono::steady_clock::now();
                    // (#3 eliminate) advance: every changed tensor predicted exactly -> the server
                    // advances by cached stride; ship just the graph_number. else the real patch.
                    const bool         advance = prefetch && dev_advance[id];
                    const enum rpc_cmd cmd     = advance ? RPC_CMD_GRAPH_ADVANCE : RPC_CMD_PATCH_VIEWS;
                    const void *       payload =
                        advance ? (const void *) &this_graph_number : (const void *) dev_patch[id].data();
                    const size_t psize = advance ? sizeof(this_graph_number) : dev_patch[id].size();
                    bool         status;
                    if (rpc_graph_oneway()) {  // fire-and-forget, pipelined before DO_COMPUTATION
                        status = send_rpc_cmd_oneway(sock, cmd, payload, psize);
                    } else {
                        status = send_rpc_cmd(sock, cmd, payload, psize, nullptr, 0);
                    }
                    g_graph_send_ns +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _t_gs)
                            .count();
                    GGML_ASSERT(status);
                });
            }
            for (auto & t : send_threads) {
                if (t.joinable()) {
                    t.join();
                }
            }
            g_send_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _t_send)
                    .count();
            if (dbg_diffcache) {
                GGML_LOG_INFO("[rpc-diffcache] patched %d tensors (across %d devices)\n", total_patches.load(),
                              device_count);
            }
        }

        //send a commend to servers to ask them do the computation job here
        auto                     _t_compute = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;

        ggml_tensor *        tensor = cgraph->nodes[cgraph->n_nodes - 1];
        std::vector<uint8_t> data(ggml_nbytes(tensor), 0);
        std::mutex           data_mutex;
        for (int id = 0; id < device_count; ++id) {
            threads.emplace_back([&, id]() {
                auto dev_ctx = (ggml_backend_rpc_device_context *) reg_ctx->devices[id]->context;
                auto sock    = get_socket(dev_ctx->endpoint);

                rpc_msg_do_computation_req compute_info;
                compute_info.graph_number = this_graph_number;
                auto _t_dc                = std::chrono::steady_clock::now();
                bool status =
                    send_rpc_cmd(sock, RPC_CMD_DO_COMPUTATION, &compute_info, sizeof(compute_info), nullptr, 0);
                g_do_comp_ns +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _t_dc)
                        .count();
                GGML_ASSERT(status);

                if (strcmp(tensor->name, "result_output") == 0) {
                    // GGML_LOG_INFO("getting result data from device %d\n", id);
                    auto _t_g = std::chrono::steady_clock::now();
                    add_data_to_data(data, tensor, data_mutex, id);
                    g_gather_ns +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _t_g)
                            .count();
                }
            });
        }
        // Join all threads
        for (auto & thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        g_compute_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _t_compute).count();
        // Write back data to RPC backend tensor on the client

        if (strcmp(tensor->name, "result_output") == 0) {
            ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
            buf->iface.set_tensor(buf, tensor, data.data(), 0, data.size());
            static const bool dbg_timing = getenv("RPC_DBG_TIMING") != nullptr;
            if (dbg_timing) {
                // log THIS forward's split (delta since last) -- the first forward is the
                // prompt eval (big execute, tiny graph-send %); decode forwards reveal the
                // real steady-state ratio. Also keep the cumulative.
                static long long prev_gs = 0;
                static long long prev_dc = 0;
                long long        gs      = g_graph_send_ns.load();
                long long        dc      = g_do_comp_ns.load();
                long long        dgs     = gs - prev_gs;
                long long        ddc     = dc - prev_dc;
                prev_gs                  = gs;
                prev_dc                  = dc;
                // per-token client-side phase WALL breakdown: where the decode token actually goes.
                // build = client CPU (build_segment_tensors + diff + verify); send = graph-send
                // (PATCH/ADVANCE); compute = DO_COMPUTATION phase wall (all-reduce + gather);
                // gather = result-gather (GET_TENSOR + sum, summed across devices, wall ~ /N).
                static long long prev_b = 0, prev_sw = 0, prev_cw = 0, prev_g = 0;
                long long        b = g_build_ns.load(), sw = g_send_ns.load();
                long long        cw = g_compute_ns.load(), gth = g_gather_ns.load();
                long long        db = b - prev_b, dsw = sw - prev_sw, dcw = cw - prev_cw, dg = gth - prev_g;
                static long long prev_bst = 0;
                long long        bst  = g_bst_ns.load();
                long long        dbst = bst - prev_bst;
                prev_bst = bst;
                prev_b  = b;
                prev_sw = sw;
                prev_cw = cw;
                prev_g  = gth;
                GGML_LOG_INFO(
                    "[rpc-timing] fwd %d: this graph_send=%.3fs execute+allreduce=%.3fs (this=%.1f%%) "
                    "| cumulative=%.1f%%\n",
                    ++g_compute_tokens, dgs / 1e9, ddc / 1e9, 100.0 * dgs / (double) (dgs + ddc + 1),
                    100.0 * gs / (double) (gs + dc + 1));
                GGML_LOG_INFO(
                    "[rpc-phase]  fwd %d: build=%.3fs (build_seg_tensors~%.3fs) send=%.3fs compute_wall=%.3fs "
                    "gather=%.3fs (all-reduce~%.3fs) [build is client CPU; *_seg/gather are /N wall]\n",
                    g_compute_tokens.load(), db / 1e9, dbst / (double) device_count / 1e9, dsw / 1e9, dcw / 1e9,
                    dg / 1e9, (dcw - dg / (double) device_count) / 1e9);
            }
        }
        return GGML_STATUS_SUCCESS;
    } else {
        // Non-split (single device / no -sm row / pipeline layer-distribution). Stock llama.cpp
        // RPC re-serializes + re-ships + re-deserializes the WHOLE graph every token. The PP diff
        // cache (opt-in RPC_PP_DIFF) ships it ONCE per topology, then per-token patches only the
        // changed tensors of the server's stored graph -- killing PP's dominant per-token cost.
        rpc_msg_graph_compute_rsp response;
        auto                      sock = get_socket(rpc_ctx->endpoint);

        static const bool pp_diff = rpc_opt_enabled() && getenv("RPC_PP_DIFF") != nullptr;
        if (!pp_diff) {
            std::vector<uint8_t> input;
            serialize_graph(cgraph, input);
            bool status =
                send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size(), &response, sizeof(response));
            GGML_ASSERT(status);
            return (enum ggml_status) response.result;
        }

        std::vector<rpc_tensor> tensors = build_graph_tensors(cgraph);
        uint64_t                hash    = rpc_graph_topo_hash(cgraph);
        static std::unordered_map<std::string, pp_diff_state> pp_states;  // per-server (endpoint) state
        pp_diff_state &                                       st = pp_states[rpc_ctx->endpoint];

        // (prefetch) stack the next-token predictor on the PP diff cache, exactly as the TP path
        // does: predict this token's array (constant per-field stride from the last delta) and, when
        // the prediction holds, ship a payload-free ADVANCE_COMPUTE -- the server advances its stored
        // graph by the cached stride + computes, zero patch bytes on the wire. DEFAULT ON with the PP
        // diff cache; opt out with RPC_NO_PREFETCH (must match the server's srv_prefetch gate).
        static const bool pp_prefetch  = getenv("RPC_NO_PREFETCH") == nullptr;
        static const bool pp_pf_dbg    = getenv("RPC_PREFETCH_DBG") != nullptr;
        auto              it           = st.pp_cache.find(hash);
        bool              hit          = (it != st.pp_cache.end()) && st.last_sent.count(it->second);
        if (hit) {
            uint8_t                   gnum = it->second;
            std::vector<rpc_tensor> & prev = st.last_sent[gnum];

            // (prefetch) verify last token's prediction against this token's real array. An ADVANCE is
            // safe only if EVERY tensor matches, so the server's stride-advance reproduces this exact
            // array; any mismatch -> fall back to a real patch (which also rebuilds the server stride).
            bool advance = false;
            if (pp_prefetch) {
                std::vector<rpc_tensor> & pred = st.last_pred[gnum];
                advance                        = (pred.size() == tensors.size() && !pred.empty());
                for (size_t i = 0; advance && i < tensors.size(); i++) {
                    const rpc_tensor & a = tensors[i];
                    const rpc_tensor & p = pred[i];
                    if (p.data != a.data || p.view_offs != a.view_offs || p.flags != a.flags ||
                        memcmp(p.ne, a.ne, sizeof(a.ne)) != 0 || memcmp(p.nb, a.nb, sizeof(a.nb)) != 0 ||
                        memcmp(p.op_params, a.op_params, sizeof(a.op_params)) != 0) {
                        advance = false;
                    }
                }
            }

            if (advance) {
                // predicted exactly -> no patch payload; server advances by cached stride + computes
                bool status =
                    send_rpc_cmd(sock, RPC_CMD_ADVANCE_COMPUTE, &gnum, sizeof(gnum), &response, sizeof(response));
                GGML_ASSERT(status);
            } else {
                // HIT: patch only the changed tensors. Payload: graph_number(1) | n_segments(4)=1 |
                //   n_patches(4) | rpc_view_patch[n_patches]  (the layout patch_views expects).
                std::vector<rpc_view_patch> patches;
                size_t                      n = std::min(tensors.size(), prev.size());
                for (size_t i = 0; i < n; i++) {
                    if (rpc_tensor_changed(tensors[i], prev[i])) {
                        rpc_view_patch p;
                        p.idx = (uint32_t) i;
                        p.t   = tensors[i];
                        patches.push_back(p);
                    }
                }
                std::vector<uint8_t> payload;
                uint32_t             n_seg = 1, n_patches = (uint32_t) patches.size();
                payload.push_back(gnum);
                payload.insert(payload.end(), (uint8_t *) &n_seg, (uint8_t *) &n_seg + sizeof(n_seg));
                payload.insert(payload.end(), (uint8_t *) &n_patches, (uint8_t *) &n_patches + sizeof(n_patches));
                payload.insert(payload.end(), (uint8_t *) patches.data(),
                               (uint8_t *) patches.data() + (size_t) n_patches * sizeof(rpc_view_patch));
                bool status = send_rpc_cmd(sock, RPC_CMD_PATCH_COMPUTE, payload.data(), payload.size(), &response,
                                           sizeof(response));
                GGML_ASSERT(status);
            }
            if (pp_pf_dbg) {
                GGML_LOG_INFO("[PP-PREFETCH] gnum=%u %s\n", gnum, advance ? "ADVANCE (no payload)" : "patch");
            }

            // (prefetch) synthesize the prediction for token+1: advance every field by its stride
            // (this token - last token). Mirrors the server's graph_advance arithmetic exactly.
            if (pp_prefetch && prev.size() == tensors.size()) {
                std::vector<rpc_tensor> & pred = st.last_pred[gnum];
                pred.resize(tensors.size());
                for (size_t i = 0; i < tensors.size(); i++) {
                    const rpc_tensor & a = tensors[i];
                    const rpc_tensor & b = prev[i];
                    pred[i]              = a;
                    for (int d = 0; d < GGML_MAX_DIMS; d++) {
                        pred[i].ne[d] = a.ne[d] + (a.ne[d] - b.ne[d]);
                        pred[i].nb[d] = a.nb[d] + (a.nb[d] - b.nb[d]);
                    }
                    for (size_t j = 0; j < GGML_MAX_OP_PARAMS / sizeof(int32_t); j++) {
                        pred[i].op_params[j] = a.op_params[j] + (a.op_params[j] - b.op_params[j]);
                    }
                    pred[i].flags     = a.flags + (a.flags - b.flags);
                    pred[i].data      = a.data + (a.data - b.data);
                    pred[i].view_offs = a.view_offs + (a.view_offs - b.view_offs);
                }
            } else if (pp_prefetch) {
                st.last_pred[gnum].clear();
            }

            prev = std::move(tensors);  // the server now holds these values
            return (enum ggml_status) response.result;
        }

        // MISS: ship the full graph + a fresh graph_number; the server stores + computes it.
        uint8_t              gnum = st.next_gnum++;
        std::vector<uint8_t> input;
        serialize_graph_from_tensors(cgraph, tensors, input);
        input.push_back(gnum);  // trailing graph_number
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE_STORE, input.data(), input.size(), &response,
                                   sizeof(response));
        GGML_ASSERT(status);
        st.pp_cache[hash]   = gnum;
        st.last_sent[gnum]  = std::move(tensors);
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

    all_reduce_block(ggml_tensor * tensor, int op, int device_count, ggml_backend_t backend, uint8_t device_id,
                     uint32_t seq);
    ~all_reduce_block();
    bool block_init(ggml_tensor * tensor, int op, int device_count, ggml_backend_t backend, uint8_t device_id,
                    uint32_t seq, bool await = false);
    bool add(std::vector<uint8_t> & input, uint8_t src_id);

    // Buffer a partial that arrived before we reached this all-reduce, keyed by its
    // sequence (token). Lets a peer run ahead without its partial being misapplied to
    // the wrong token or lost on reset -- needed once the graph cache removes the
    // GRAPH_COMPUTE round-trips that used to implicitly barrier the peers. The source
    // device id is kept so the deterministic fold can place it in the right slot.
    bool add_to_buffer(uint32_t seq, uint8_t src_id, const std::vector<uint8_t> & input) {
        std::lock_guard<std::mutex> lock(add_mutex);
        all_reduce_buffer[seq].push_back({ src_id, input });
        return true;
    }

    // TREE all-reduce, NON-ROOT side. This server sent its partial to the root and now waits
    // for the single folded result; set_result writes it into `tensor` and completes the wait.
    // (No fold here -- the root already folded in ascending-id order, so this is bit-identical
    // to all-to-all and the same f32 result on every server.) An early result (arrived before
    // we block_init for this token) is buffered by seq and applied on init.
    bool set_result(const std::vector<uint8_t> & result) {
        std::lock_guard<std::mutex> lock(add_mutex);
        ggml_backend_tensor_set(tensor, result.data(), 0, result.size());
        {
            std::lock_guard<std::mutex> wlock(wait_mutex);
            is_completed = true;
            initialized  = false;
        }
        cv.notify_all();
        return true;
    }

    bool add_result_to_buffer(uint32_t seq, const std::vector<uint8_t> & result) {
        std::lock_guard<std::mutex> lock(add_mutex);
        result_buffer[seq] = result;
        return true;
    }

    uint32_t get_current_seq() const { return current_seq; }

    //reset the block for the next-time use
    void block_uinit() {
        initialized  = false;
        arrived      = 0;
        is_completed = false;
        slots.clear();
        ggml_backend_buffer_free(add_tensor->buffer);
        ggml_free(ctx);
    }

    bool is_init() { return initialized; }

    bool wait_for_completion();

  private:
    // reduced-precision partials (RPC_AR_PARTIAL): round-trip this server's own partial (slot
    // self_id) through the chosen wire format so it matches the rounded copies the peers received
    // -> the ordered fold is bit-identical on every server. No-op when RPC_AR_PARTIAL=f32 (or the
    // reduced tensor isn't f32).
    void ar_fp16_roundtrip_self() {
        const rpc_ar_fmt fmt = rpc_ar_partial();
        if (fmt == rpc_ar_fmt::f32 || tensor->type != GGML_TYPE_F32 || self_id < 0) {
            return;
        }
        int64_t n = (int64_t) (reduce_nbytes / sizeof(float));
        float * x = (float *) slots[self_id].data();
        if (fmt == rpc_ar_fmt::f16) {
            std::vector<ggml_fp16_t> h(n);
            ggml_fp32_to_fp16_row(x, h.data(), n);
            ggml_fp16_to_fp32_row(h.data(), x, n);
        } else if (fmt == rpc_ar_fmt::e4m3) {  // fp8: round-trip self to match the fp8 copies peers got
            std::vector<uint8_t> q(n);
            rpc_e4m3_quantize(x, q.data(), n);
            rpc_e4m3_dequantize(q.data(), x, n);
        } else if (fmt == rpc_ar_fmt::i8b) {  // per-block int8: round-trip self to match peers' copies
            std::vector<uint8_t> q(rpc_i8b_bytes(n));
            rpc_i8b_quantize(x, q.data(), n);
            rpc_i8b_dequantize(q.data(), x, n);
        } else {  // i8: quantize + dequantize so self matches the int8-rounded copies peers got
            std::vector<int8_t> q(n);
            const float         scale = rpc_i8_quantize(x, q.data(), n);
            rpc_i8_dequantize(q.data(), scale, x, n);
        }
    }

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
    int                               arrived      = 0;       //number of servers whose partial has arrived
    // Deterministic fold: collect every server's partial in a per-source slot and sum them
    // in ascending device-id order (== ascending contraction-K-slice order == what a single
    // device's contiguous-K matmul reduction does), independent of network arrival order.
    // Without this the F32 sum order races on arrival and N>=3 output varies run-to-run.
    std::vector<std::vector<uint8_t>> slots;               //slots[d] = device d's partial bytes
    int                               self_id       = -1;  //this server's device id (its own slot)
    size_t                            reduce_nbytes = 0;   //bytes per partial
    std::unordered_map<uint32_t, std::vector<std::pair<uint8_t, std::vector<uint8_t>>>>
         all_reduce_buffer;                                //seq(token) -> (src_id, partial) that arrived before init
    // TREE all-reduce non-root: await the root's result instead of folding. result_buffer holds
    // a result that arrived before we reached this token's all-reduce (applied on block_init).
    bool await_result = false;
    std::unordered_map<uint32_t, std::vector<uint8_t>> result_buffer;  //seq -> root's result (early)
    uint32_t              current_seq = 0;                             //the sequence (token) this block is reducing now
    ggml_backend_t        backend;                                     //backend type
    struct ggml_context * ctx;
};

bool all_reduce_block::wait_for_completion() {
    std::unique_lock<std::mutex> lock(wait_mutex);
    // Bounded wait: a lost partial over lossy wifi must not deadlock forever.
    // Return false on timeout so the caller aborts the compute cleanly (run is
    // retriable) instead of hanging. Override seconds via RPC_ALLREDUCE_TIMEOUT.
    static const int timeout_s = getenv("RPC_ALLREDUCE_TIMEOUT") ? atoi(getenv("RPC_ALLREDUCE_TIMEOUT")) : 30;
    return cv.wait_for(lock, std::chrono::seconds(timeout_s), [this] { return is_completed; });
}

//function description: create the all_reduce_block for a specific tensor and initialize it by the current server, create the add tensor and graph
//when to be called:    when the current server finishes graph computing
//note:                 since the first time to be called, no data in the buffer
all_reduce_block::all_reduce_block(ggml_tensor * tensor, int op, int device_count, ggml_backend_t backend,
                                   uint8_t device_id, uint32_t seq) :
    op(op),
    num_of_servers(device_count),
    backend(backend) {
    // GGML_LOG_INFO("all_reduce_block called\n");
    current_seq  = seq;  // first all-reduce of this tensor on this server -> token `seq`
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

    // OPTIMIZED: capture this server's own partial into its device-id slot; the fold sums all
    // slots in ascending id order once every server's partial has arrived (deterministic).
    // BASELINE (RPC_NO_OPT): no slots -- `tensor` already holds P_self and add() accumulates
    // peers into it in arrival order (nondeterministic at N>=3, the pre-optimization behavior).
    if (rpc_opt_enabled()) {
        reduce_nbytes = ggml_nbytes(tensor);
        self_id       = device_id;
        slots.assign(num_of_servers, {});
        slots[device_id].resize(reduce_nbytes);
        ggml_backend_tensor_get(tensor, slots[device_id].data(), 0, reduce_nbytes);
        ar_fp16_roundtrip_self();  // fp16: round-trip self so all servers fold identical slots
    }
    arrived = 1;

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
                                  uint8_t device_id, uint32_t seq, bool await) {
    // GGML_LOG_INFO("all_reduce_block init called\n");
    if (initialized) {
        return true;
    }
    current_seq = seq;  // this block now reduces token `seq`; only matching partials apply

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
    initialized  = true;
    await_result = await;

    if (await) {
        // TREE non-root: we sent our partial to the root and now just await its result. No
        // self-capture / fold. Apply a result that arrived early (buffered by seq) if any.
        auto rit = result_buffer.find(current_seq);
        if (rit != result_buffer.end()) {
            std::vector<uint8_t> r = std::move(rit->second);
            result_buffer.erase(rit);
            set_result(r);
        }
        return true;
    }

    // OPTIMIZED: capture self's partial into its device-id slot (see ctor). BASELINE: skip --
    // `tensor` holds P_self and add() folds peers in arrival order. arrived starts at 1 (self).
    if (rpc_opt_enabled()) {
        reduce_nbytes = ggml_nbytes(tensor);
        self_id       = device_id;
        slots.assign(num_of_servers, {});
        slots[device_id].resize(reduce_nbytes);
        ggml_backend_tensor_get(tensor, slots[device_id].data(), 0, reduce_nbytes);
        ar_fp16_roundtrip_self();  // fp16: round-trip self so all servers fold identical slots
    }
    arrived = 1;

    //apply any partials that arrived early for THIS sequence (token); leave others buffered
    auto buf_it = all_reduce_buffer.find(current_seq);
    if (buf_it != all_reduce_buffer.end()) {
        for (auto & part : buf_it->second) {
            add(part.second, part.first);  // (src_id, bytes) -> place in the right slot
        }
        all_reduce_buffer.erase(buf_it);
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

bool all_reduce_block::add(std::vector<uint8_t> & input, uint8_t src_id) {
    // GGML_LOG_INFO("all_reduce_block add called\n");
    std::lock_guard<std::mutex> lock(add_mutex);

    if (!rpc_opt_enabled()) {
        // BASELINE: arrival-order fold -- add this peer's partial into `tensor` immediately
        // (tensor += add_tensor). FP sum order = arrival order => nondeterministic at N>=3.
        // This is the pre-optimization behavior.
        ggml_backend_tensor_set(add_tensor, input.data(), 0, input.size());
        ggml_status result = ggml_backend_graph_compute(backend, graph);
        if (result != GGML_STATUS_SUCCESS) {
            GGML_LOG_INFO("graph_compute failed\n");
        }
        arrived++;
        if (arrived >= num_of_servers) {
            {
                std::lock_guard<std::mutex> wlock(wait_mutex);
                is_completed = true;
                initialized  = false;
            }
            cv.notify_all();
        }
        GGML_UNUSED(src_id);
        return is_completed;
    }

    // OPTIMIZED: store this peer's partial in its device-id slot (don't sum yet). Counting
    // only the first arrival per source keeps arrived correct even if a duplicate slips in.
    // fp16: `input` arrived as f16 -- upcast to f32 into the slot so the fold stays f32.
    if (src_id < slots.size()) {
        if (slots[src_id].empty()) {
            arrived++;
        }
        const rpc_ar_fmt fmt = rpc_ar_partial();
        if (fmt != rpc_ar_fmt::f32 && tensor->type == GGML_TYPE_F32) {
            int64_t              n = (int64_t) (reduce_nbytes / sizeof(float));
            std::vector<uint8_t> f32(reduce_nbytes);
            if (fmt == rpc_ar_fmt::f16) {          // arrived as f16 -- upcast to f32 into the slot
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) input.data(), (float *) f32.data(), n);
            } else if (fmt == rpc_ar_fmt::e4m3) {  // arrived as fp8 e4m3[n] -- dequantize to f32
                rpc_e4m3_dequantize((const uint8_t *) input.data(), (float *) f32.data(), n);
            } else if (fmt == rpc_ar_fmt::i8b) {   // arrived as per-block int8 -- dequantize to f32
                rpc_i8b_dequantize((const uint8_t *) input.data(), (float *) f32.data(), n);
            } else {                               // i8: arrived as scale(f32) | int8[n] -- dequantize to f32
                float scale;
                memcpy(&scale, input.data(), sizeof(float));
                rpc_i8_dequantize((const int8_t *) (input.data() + sizeof(float)), scale, (float *) f32.data(), n);
            }
            slots[src_id] = std::move(f32);
        } else {
            slots[src_id] = input;
        }
    }

    // Once every server's partial is in, fold them in ASCENDING device-id order (slot 0,
    // 1, ... N-1). FP addition isn't associative, so a fixed order is what makes the result
    // reproducible run-to-run AND bit-identical on every server (all hold the same slots).
    if (arrived >= num_of_servers) {
        bool first = true;
        for (int d = 0; d < num_of_servers; d++) {
            if (slots[d].empty()) {
                continue;
            }
            if (first) {
                ggml_backend_tensor_set(tensor, slots[d].data(), 0, slots[d].size());  // tensor = slots[d]
                first = false;
            } else {
                ggml_backend_tensor_set(add_tensor, slots[d].data(), 0, slots[d].size());
                ggml_status result = ggml_backend_graph_compute(backend, graph);  // tensor += add_tensor
                if (result != GGML_STATUS_SUCCESS) {
                    GGML_LOG_INFO("graph_compute failed\n");
                }
            }
        }

        //notify the thread in wait_for_completion. Set the predicate UNDER wait_mutex so the
        //wakeup can't be lost (a lost wakeup stalls for the full RPC_ALLREDUCE_TIMEOUT).
        {
            std::lock_guard<std::mutex> wlock(wait_mutex);
            is_completed = true;
            initialized  = false;
        }
        cv.notify_all();
    }
    return is_completed;
}

// (#3 prefetch=eliminate) per-tensor signed stride (this patch - previous patch). When the
// client predicts a token's patch exactly (steady state), it sends RPC_CMD_GRAPH_ADVANCE with
// NO payload and the server applies tensor += stride -- zero graph-send bytes on the wire.
struct tensor_stride {
    int64_t ne[GGML_MAX_DIMS]{};
    int64_t nb[GGML_MAX_DIMS]{};
    int64_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]{};
    int64_t flags{};
    int64_t data{};
    int64_t view_offs{};
    bool    valid = false;
};

struct graph_info {
    uint8_t                    graph_number;
    ggml_cgraph *              cgraph;
    ggml_context *             ctx;
    uint8_t                    signal;
    graph_info *               next = nullptr;
    // Diff cache: the ggml_tensor created for the i-th rpc_tensor the client sent for
    // this segment (same index the client patches by). Lets RPC_CMD_PATCH_VIEWS update
    // a stored tensor's view_offs/data in place without re-deserializing the graph.
    std::vector<ggml_tensor *> by_idx;
    // (#3 prefetch) by_idx-aligned stride set when a real patch applies, + the idx set from
    // that patch (the tensors RPC_CMD_GRAPH_ADVANCE advances).
    std::vector<tensor_stride> strides;
    std::vector<uint32_t>      last_patched;
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
    bool graph_compute_batch(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool set_split(rpc_msg_set_split_rsp & response);
    bool create_peer_connection(const rpc_msg_create_peer_connection_req & request,
                                rpc_msg_create_peer_connection_rsp &       response);
    bool send_to_peer(const rpc_msg_send_to_peer_req & request, rpc_msg_send_to_peer_rsp & response);
    void add_socket_listen(const std::shared_ptr<socket_t> & sock);
    bool all_reduce(std::vector<uint8_t> & input);
    bool ar_result(std::vector<uint8_t> & input);  // tree all-reduce: non-root applies root's result
    bool do_computation(const rpc_msg_do_computation_req & request);
    bool patch_views(const std::vector<uint8_t> & input);
    // (PP diff cache) non-split store-then-patch, computed inline (no DO_COMPUTATION / all-reduce):
    bool graph_compute_store(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response);
    bool patch_compute(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response);
    bool advance_compute(uint8_t graph_number, rpc_msg_graph_compute_rsp & response);
    bool graph_advance(uint8_t graph_number);
    bool load_cached(const rpc_msg_load_cached_req & request, rpc_msg_load_cached_rsp & response);
    bool set_tensor_cache(const std::vector<uint8_t> & input);
    bool batch_load_cached(const std::vector<uint8_t> & input, std::vector<uint8_t> & response);

    ggml_backend_t & get_backend() { return backend; }
  private:
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id, struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor *> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor *> &     tensor_map);
    void store_graph_compute_info(uint8_t graph_number, ggml_cgraph * cgraph, ggml_context * ctx, uint8_t signal,
                                  std::vector<ggml_tensor *> && by_idx);
    ggml_backend_t                                           backend;
    std::unordered_set<ggml_backend_buffer_t>                buffers;
    bool                                                     server_split = false;
    std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets_connectto;  //sockets that the server connects to
    // Hold a shared_ptr to each dialed peer socket so the weak_ptrs above don't
    // expire. Otherwise create_peer_connection's local shared_ptr is released
    // immediately and EVERY all-reduce re-dials its peers (a TCP handshake per
    // peer, per layer, per token) -- the dominant decode cost over WiFi.
    std::vector<std::shared_ptr<socket_t>>                   peer_socks_held;
    std::vector<std::string> peer_endpoints;  //device id -> endpoint (for tree all-reduce: find the root, device 0)
    std::vector<std::weak_ptr<socket_t>>                sockets_listento;     //sockets that the server listen to
    std::mutex                                          sockets_mutex;        //mutex for adding sockets to the list
    uint8_t                                             device_id;            //device id for current server
    uint8_t                                             device_count;         //total num of servers
    std::unordered_map<std::string, all_reduce_block *> all_reduce_blocks;    //blocks for all reduce
    std::unordered_map<std::string, uint32_t>           all_reduce_seq;       //per-tensor all-reduce sequence (token)
    std::mutex                                          block_mutex;          //mutex for adding or checking blocks
    std::unordered_map<uint8_t, graph_compute_info *>   graph_compute_infos;  // map graph_number to graph_compute_info
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
        // require that the tensor data does not go beyond the buffer end.
        // Empty tensors (any ne==0 -- e.g. a 0-token V-cache write view on a
        // pruned/warmup ubatch, which the split path intentionally carries through)
        // occupy no bytes, but ggml_nbytes UNDERFLOWS for them ((ne[i]-1)*nb[i]
        // wraps to ~2^64), tripping this OOB assert. Treat their size as 0 -- they
        // carry no data and the server skips their compute.
        uint64_t tensor_size  = ggml_is_empty(result) ? 0 : (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size  = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        // Diagnostic for the intermittent OOB abort: dump the exact offending tensor
        // (name/op/shape/view_offs/data vs buffer bounds + how far out) before aborting,
        // so a single crash pinpoints which remapped tensor produced the bad pointer.
        bool     oob_overflow = !(tensor->data + tensor_size >= tensor->data);
        bool     oob_below    = tensor->data < buffer_start;
        bool     oob_above    = tensor->data + tensor_size > buffer_start + buffer_size;
        if (oob_overflow || oob_below || oob_above) {
            GGML_LOG_ERROR(
                "[deserialize_tensor] OOB '%s' op=%d type=%d ne=[%lld,%lld,%lld,%lld] "
                "nb=[%u,%u,%u,%u] view_offs=%llu data=0x%llx size=%llu buffer=0x%llx start=0x%llx bytes=%llu "
                "| overflow=%d below=%d above=%d under_by=%lld over_by=%lld\n",
                tensor->name, (int) tensor->op, (int) tensor->type, (long long) result->ne[0],
                (long long) result->ne[1], (long long) result->ne[2], (long long) result->ne[3], tensor->nb[0],
                tensor->nb[1], tensor->nb[2], tensor->nb[3], (unsigned long long) tensor->view_offs,
                (unsigned long long) tensor->data, (unsigned long long) tensor_size,
                (unsigned long long) tensor->buffer, (unsigned long long) buffer_start,
                (unsigned long long) buffer_size, oob_overflow, oob_below, oob_above,
                (long long) (buffer_start - tensor->data),
                (long long) ((tensor->data + tensor_size) - (buffer_start + buffer_size)));
        }
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
    static const bool on = rpc_opt_enabled() && (getenv("RPC_NO_WEIGHT_CACHE") == nullptr);
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

namespace fs = std::filesystem;

// Cache size cap in bytes (RPC_WEIGHT_CACHE_MAX_GB, default 8 GB; 0 = unbounded). The cache is
// content-addressed and otherwise never self-cleaned, so it grows forever across models/shardings.
static uint64_t rpc_weight_cache_max_bytes() {
    static const uint64_t max_b = [] {
        const char * env = getenv("RPC_WEIGHT_CACHE_MAX_GB");
        double       gb  = env ? atof(env) : 8.0;
        return gb > 0 ? (uint64_t) (gb * 1e9) : (uint64_t) 0;
    }();
    return max_b;
}

// LRU eviction: if the cache dir exceeds the cap, delete oldest-mtime files until under it.
// mtime is refreshed on every HIT (load_cached), so "oldest" == least-recently-USED and the
// model being loaded now (freshly stored/hit) is never evicted. Skips .tmp.* (in-flight atomic
// writes). Best-effort -- all errors ignored; one pruner at a time (concurrent stores skip).
static void rpc_weight_cache_prune() {
    const uint64_t max_b = rpc_weight_cache_max_bytes();
    if (max_b == 0 || !rpc_weight_cache_enabled()) {
        return;
    }
    static std::mutex            prune_mtx;
    std::unique_lock<std::mutex> lk(prune_mtx, std::try_to_lock);
    if (!lk.owns_lock()) {
        return;  // another thread is already pruning
    }
    struct centry { fs::file_time_type t; uintmax_t sz; fs::path p; };
    std::vector<centry> files;
    uint64_t            total = 0;
    std::error_code     ec;
    fs::directory_iterator it(rpc_weight_cache_dir(), ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const fs::path & p = it->path();
        if (p.filename().string().find(".tmp.") != std::string::npos) {
            continue;  // in-flight atomic write, not a published entry
        }
        std::error_code    e2;
        uintmax_t          sz = fs::file_size(p, e2);
        fs::file_time_type t  = fs::last_write_time(p, e2);
        if (e2) {
            continue;
        }
        files.push_back({ t, sz, p });
        total += sz;
    }
    if (total <= max_b) {
        return;
    }
    std::sort(files.begin(), files.end(), [](const centry & a, const centry & b) { return a.t < b.t; });  // oldest first
    static const bool dbg = getenv("RPC_DBG_WCACHE") != nullptr;
    for (const centry & f : files) {
        if (total <= max_b) {
            break;
        }
        std::error_code e3;
        if (fs::remove(f.p, e3)) {
            total -= f.sz;
            if (dbg) {
                GGML_LOG_INFO("[wcache] EVICT %s (%llu B)\n", f.p.filename().string().c_str(), (unsigned long long) f.sz);
            }
        }
    }
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
    // Refresh mtime so LRU eviction counts this as recently used (never evict what's loading now).
#ifndef _WIN32
    utime(rpc_weight_cache_path(request.hash).c_str(), nullptr);
#endif
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

    // Persist to the on-disk cache (best-effort; a failed write just misses next run).
    // ATOMIC publish: write a unique temp file, then rename() into place. rename(2) is atomic
    // on the same filesystem, so a concurrent LOAD_CACHED reader sees either the old complete
    // file or the fully-written new one -- never a partial. Two clients writing the same key
    // write identical (content-addressed) bytes, so the last rename wins and both are valid --
    // no lock needed. The temp name is unique per (process, write): the ASLR'd address of a
    // static salts the process, an atomic counter salts the write, so concurrent writers (even
    // two server processes sharing a dir on localhost) never clobber each other's temp.
    if (rpc_weight_cache_enabled()) {
        static std::atomic<uint64_t> wc_seq{ 0 };
        const std::string            path = rpc_weight_cache_path(hash);
        const std::string            tmp  = path + ".tmp." +
                                 std::to_string((unsigned long long) (uintptr_t) &wc_seq) + "." +
                                 std::to_string((unsigned long long) wc_seq.fetch_add(1));
        bool ok = false;
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (out) {
                out.write((const char *) data, (std::streamsize) size);
                ok = out.good();
            }
        }  // flush + close before the rename
        if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
            std::remove(tmp.c_str());  // write or rename failed -> drop the temp; miss next run
        }
        static const bool dbg_wcache = (getenv("RPC_DBG_WCACHE") != nullptr);
        if (dbg_wcache) {
            GGML_LOG_INFO("[wcache] STORE %016llx (%zu bytes)\n", (unsigned long long) hash, size);
        }
        // Bound the cache: after ~cap/4 bytes have been stored, scan+evict down to the cap.
        // Size-driven (not store-count) so it works regardless of model/tensor count, and the
        // overshoot is bounded to ~cap/4 between prunes.
        const uint64_t cap = rpc_weight_cache_max_bytes();
        if (cap) {
            static std::atomic<uint64_t> since_prune{ 0 };
            if (since_prune.fetch_add(size) + size >= cap / 4) {
                since_prune.store(0);
                rpc_weight_cache_prune();
            }
        }
    }
    return true;
}

// BATCH_LOAD_CACHED: | n(4) | { rpc_tensor | key(8) } * n |. For each entry, load the cache HIT
// from local disk straight into its buffer and set response[i]=1; a miss leaves 0 (the client
// uploads it). Parallelized across a small thread pool -- entries write distinct buffer regions,
// so no locking. Replaces ~n sequential per-tensor LOAD_CACHED round-trips with one message.
bool rpc_server::batch_load_cached(const std::vector<uint8_t> & input, std::vector<uint8_t> & response) {
    if (input.size() < sizeof(uint32_t)) {
        return false;
    }
    uint32_t n;
    memcpy(&n, input.data(), sizeof(n));
    const size_t entry = sizeof(rpc_tensor) + sizeof(uint64_t);
    if (input.size() != sizeof(uint32_t) + (size_t) n * entry) {
        return false;
    }
    response.assign(n, 0);
    if (n == 0 || !rpc_weight_cache_enabled()) {
        return true;  // all miss -> client uploads
    }
    const uint8_t * base   = input.data() + sizeof(uint32_t);
    auto            worker = [&](uint32_t lo, uint32_t hi) {
        for (uint32_t i = lo; i < hi; ++i) {
            const uint8_t *    p  = base + (size_t) i * entry;
            const rpc_tensor * rt = (const rpc_tensor *) p;
            uint64_t           key;
            memcpy(&key, p + sizeof(rpc_tensor), sizeof(key));
            const std::string path = rpc_weight_cache_path(key);
            std::ifstream     f(path, std::ios::binary | std::ios::ate);
            if (!f) {
                continue;  // miss
            }
            const std::streamsize sz = f.tellg();
            if (sz <= 0) {
                continue;
            }
            f.seekg(0, std::ios::beg);
            std::vector<uint8_t> data((size_t) sz);
            if (!f.read((char *) data.data(), sz)) {
                continue;
            }
            struct ggml_init_params params{ ggml_tensor_overhead(), NULL, true };
            struct ggml_context *   ctx    = ggml_init(params);
            ggml_tensor *           tensor = deserialize_tensor(ctx, rt);
            if (tensor != nullptr && (size_t) sz == ggml_nbytes(tensor)) {
                ggml_backend_tensor_set(tensor, data.data(), 0, (size_t) sz);  // distinct region per entry
                response[i] = 1;                                               // hit
#ifndef _WIN32
                utime(path.c_str(), nullptr);  // LRU touch
#endif
            }
            ggml_free(ctx);
        }
    };
    const unsigned hw       = std::thread::hardware_concurrency();
    const unsigned nthreads = std::min<unsigned>(hw ? hw : 4u, 4u);
    if (n <= 1 || nthreads <= 1) {
        worker(0, n);
    } else {
        std::vector<std::thread> pool;
        const uint32_t           chunk = (n + nthreads - 1) / nthreads;
        for (unsigned t = 0; t < nthreads; ++t) {
            const uint32_t lo = (uint32_t) t * chunk;
            const uint32_t hi = std::min<uint32_t>(n, lo + chunk);
            if (lo < hi) {
                pool.emplace_back(worker, lo, hi);
            }
        }
        for (auto & th : pool) {
            th.join();
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

    // A zero-size set (empty tensor from n_outputs pruning) has nothing to write and may carry a
    // null/empty buffer that would trip the bounds check below; treat it as a no-op.
    if (size == 0) {
        ggml_free(ctx);
        return true;
    }

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

    // A zero-size gather (e.g. an n_outputs==0 ubatch's empty result_output, ne[1]==0) has nothing
    // to read; its tensor may legitimately have a null/empty buffer (p0==p1==0) which would otherwise
    // trip the bounds check below. Return an empty response -- the client asserts data.size() ==
    // ggml_nbytes(tensor) == 0, so this matches.
    if (request.size == 0) {
        ggml_free(ctx);
        return true;
    }

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 || request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] OOB name='%s' op=%d ne=[%lld,%lld,%lld,%lld] data=0x%llx offset=%llu size=%llu "
                           "view_offs=%llu p0=0x%llx p1=0x%llx\n",
                           __func__, request.tensor.name, (int) request.tensor.op,
                           (long long) request.tensor.ne[0], (long long) request.tensor.ne[1],
                           (long long) request.tensor.ne[2], (long long) request.tensor.ne[3],
                           (unsigned long long) request.tensor.data, (unsigned long long) request.offset,
                           (unsigned long long) request.size, (unsigned long long) request.tensor.view_offs,
                           (unsigned long long) p0, (unsigned long long) p1);
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

// Direct pipeline handoff: read our own local `src` bytes and push them straight into the
// DST peer's buffer via a normal SET_TENSOR over the peer connection we already hold
// (sockets_connectto[dst_endpoint], dialed by create_peer_connection). Bypasses the client
// relay (src -> client host -> dst). SET_TENSOR is acked by the peer, so send_rpc_cmd returns
// only once the write has landed -- the handoff stays synchronous, so the data is in place
// before the client issues the next stage's compute (which travels a DIFFERENT socket, so we
// cannot lean on TCP ordering). result=0 makes the client fall back to the relay (always safe).
bool rpc_server::send_to_peer(const rpc_msg_send_to_peer_req & request, rpc_msg_send_to_peer_rsp & response) {
    static const bool dbg = getenv("RPC_DBG_HANDOFF") != nullptr;
    response.result = 0;
    std::shared_ptr<socket_t> peer;
    {
        auto it = sockets_connectto.find(request.dst_endpoint);
        if (it != sockets_connectto.end()) {
            peer = it->second.lock();
        }
    }
    if (!peer) {
        if (dbg) { GGML_LOG_INFO("[send_to_peer] no peer link to %s -> client relay\n", request.dst_endpoint); }
        return true;  // no peer link -> result=0 -> client relays instead
    }
    struct ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor *         src = deserialize_tensor(ctx, &request.src);
    if (src == nullptr) {
        ggml_free(ctx);
        return true;  // result=0 -> fall back
    }
    const size_t size = ggml_is_empty(src) ? 0 : (size_t) ggml_nbytes(src);
    // standard SET_TENSOR payload for the DST peer: | rpc_tensor(dst) | offset(8)=0 | data |
    std::vector<uint8_t> input(sizeof(rpc_tensor) + sizeof(uint64_t) + size);
    memcpy(input.data(), &request.dst, sizeof(rpc_tensor));
    const uint64_t offset = 0;
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    if (size > 0) {
        ggml_backend_tensor_get(src, input.data() + sizeof(rpc_tensor) + sizeof(offset), 0, size);
    }
    ggml_free(ctx);
    bool ok = send_rpc_cmd(peer, RPC_CMD_SET_TENSOR, input.data(), input.size(), nullptr, 0);
    if (dbg) { GGML_LOG_INFO("[send_to_peer] pushed %zu B -> %s ok=%d\n", size, request.dst_endpoint, (int) ok); }
    response.result = ok ? 1 : 0;
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
        GGML_LOG_ERROR("[%s] tensor %llu not found in tensor_ptrs: %s\n", __func__, (unsigned long long) id, e.what());
        return nullptr;
    }
}

void rpc_server::store_graph_compute_info(uint8_t graph_number, ggml_cgraph * cgraph, ggml_context * ctx,
                                          uint8_t signal, std::vector<ggml_tensor *> && by_idx) {
    auto it = graph_compute_infos.find(graph_number);
    if (it == graph_compute_infos.end()) {
        graph_compute_info * info         = new graph_compute_info(graph_number, cgraph, ctx, signal);
        graph_compute_infos[graph_number] = info;
        it                                = graph_compute_infos.find(graph_number);
    } else {
        it->second->add_info(cgraph, ctx, signal);
    }
    // the segment just stored is the chain tail; attach its array-index -> tensor map
    it->second->tail->by_idx = std::move(by_idx);
}

// (PP diff cache) Non-split MISS: deserialize the full graph (stock format + a trailing graph_number
// byte), build the patch index, STORE it (replacing any prior graph under this number), and compute
// it inline. Unlike the stock non-split graph_compute, the ctx is kept ALIVE for later PATCH_COMPUTE.
bool rpc_server::graph_compute_store(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response) {
    if (input.size() < sizeof(uint32_t) + sizeof(uint8_t)) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, input.data(), sizeof(n_nodes));
    size_t base = sizeof(uint32_t) + (size_t) n_nodes * sizeof(uint64_t);
    if (input.size() < base + sizeof(uint32_t) + sizeof(uint8_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *) (input.data() + sizeof(n_nodes));
    uint32_t         n_tensors;
    memcpy(&n_tensors, input.data() + base, sizeof(n_tensors));
    size_t end = base + sizeof(uint32_t) + (size_t) n_tensors * sizeof(rpc_tensor);
    if (input.size() < end + sizeof(uint8_t)) {
        return false;
    }
    const rpc_tensor * tensors      = (const rpc_tensor *) (input.data() + base + sizeof(n_tensors));
    uint8_t            graph_number = input[end];  // trailing byte

    size_t buf_size = (ggml_tensor_overhead() * (n_nodes + n_tensors)) + ggml_graph_overhead_custom(n_nodes, false);
    struct ggml_init_params params = { buf_size, NULL, true };
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
        GGML_LOG_ERROR("[%s] node creation failed: %s\n", __func__, e.what());
        ggml_free(ctx);
        return false;
    }
    std::vector<ggml_tensor *> by_idx(n_tensors, nullptr);
    for (uint32_t i = 0; i < n_tensors; i++) {
        auto mit = tensor_map.find(tensors[i].id);
        if (mit != tensor_map.end()) {
            by_idx[i] = mit->second;
        }
    }
    auto old = graph_compute_infos.find(graph_number);  // replace any prior graph under this number
    if (old != graph_compute_infos.end()) {
        delete old->second;
        graph_compute_infos.erase(old);
    }
    store_graph_compute_info(graph_number, graph, ctx, /*signal=*/0, std::move(by_idx));
    response.result = ggml_backend_graph_compute(backend, graph);  // compute the just-stored graph inline
    return true;                                                   // ctx stays alive in graph_compute_infos
}

// (PP diff cache) Non-split HIT: patch the stored graph's changed tensors (by index), compute inline.
bool rpc_server::patch_compute(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response) {
    if (input.empty() || !patch_views(input)) {  // patch_views reads graph_number = input[0]
        return false;
    }
    auto it = graph_compute_infos.find(input[0]);
    if (it == graph_compute_infos.end() || it->second->head == nullptr) {
        return false;
    }
    response.result = ggml_backend_graph_compute(backend, it->second->head->cgraph);
    return true;
}

// (PP diff cache + prefetch) Non-split predicted HIT: the client predicted this token's array
// exactly, so it sent NO patch -- advance the stored graph by the per-tensor stride the last
// PATCH_COMPUTE recorded (graph_advance), then compute inline. Mirrors patch_compute minus the
// patch apply. The stride table is rebuilt on the next real PATCH_COMPUTE (a prediction miss).
bool rpc_server::advance_compute(uint8_t graph_number, rpc_msg_graph_compute_rsp & response) {
    if (!graph_advance(graph_number)) {
        return false;
    }
    auto it = graph_compute_infos.find(graph_number);
    if (it == graph_compute_infos.end() || it->second->head == nullptr) {
        return false;
    }
    response.result = ggml_backend_graph_compute(backend, it->second->head->cgraph);
    return true;
}

// Store a token's segment-graphs delivered in ONE round-trip:
//   n_segments(4) | per segment: seg_len(4) | seg_data (exact per-segment payload).
// Each segment is stored exactly as the per-segment GRAPH_COMPUTE path would.
bool rpc_server::graph_compute_batch(const std::vector<uint8_t> & input, rpc_msg_graph_compute_rsp & response) {
    if (input.size() < sizeof(uint32_t)) {
        return false;
    }
    uint32_t n_segments;
    memcpy(&n_segments, input.data(), sizeof(n_segments));
    size_t off = sizeof(uint32_t);
    for (uint32_t s = 0; s < n_segments; s++) {
        if (off + sizeof(uint32_t) > input.size()) {
            return false;
        }
        uint32_t seg_len;
        memcpy(&seg_len, input.data() + off, sizeof(seg_len));
        off += sizeof(uint32_t);
        if (off + seg_len > input.size()) {
            return false;
        }
        std::vector<uint8_t> seg(input.begin() + off, input.begin() + off + seg_len);
        off += seg_len;
        if (!graph_compute(seg, response)) {
            return false;
        }
    }
    response.result = GGML_STATUS_SUCCESS;
    return true;
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

    // Diff cache: record, in the client's sent order, the ggml_tensor created for each
    // rpc_tensor. A later RPC_CMD_PATCH_VIEWS addresses tensors by this index to patch
    // their advancing view_offs/data without re-shipping the whole graph.
    std::vector<ggml_tensor *> by_idx(n_tensors, nullptr);
    for (uint32_t i = 0; i < n_tensors; i++) {
        auto mit = tensor_map.find(tensors[i].id);
        if (mit != tensor_map.end()) {
            by_idx[i] = mit->second;
        }
    }

    //store graph compute info
    store_graph_compute_info(graph_number, graph, ctx, signal, std::move(by_idx));
    static const bool dbg_store = getenv("RPC_DBG_DIFFCACHE") != nullptr;  // gate per-MISS lifecycle spam
    if (dbg_store) {
        GGML_LOG_INFO("stored graph compute info for graph number %d\n", graph_number);
    }
    response.result = GGML_STATUS_SUCCESS;
    return true;
}

static void compare_node(ggml_tensor * node, ggml_tensor * node_to_compare) {
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

static void compare_two_graph(ggml_cgraph * graph, ggml_cgraph * graph_to_compare) {
    for (int i = 0; i < graph->n_nodes; i++) {
        compare_node(graph->nodes[i], graph_to_compare->nodes[i]);
    }
}

// Diff cache (RPC_CMD_PATCH_VIEWS): the client found this token's graph structurally
// identical to one we already stored, so instead of re-shipping the whole graph it sends
// only the changed view tensors (advancing KV-cache write positions). Patch them into the
// stored graph in place, addressed by the array index recorded at store time, then the
// following DO_COMPUTATION re-runs the patched graph. No re-deserialize of the structure.
// Payload: graph_number(1) | n_segments(4) | per segment: n_patches(4) | rpc_view_patch[].
bool rpc_server::patch_views(const std::vector<uint8_t> & input) {
    if (input.size() < sizeof(uint8_t) + sizeof(uint32_t)) {
        return false;
    }
    size_t  off          = 0;
    uint8_t graph_number = input[off];
    off += sizeof(uint8_t);
    uint32_t n_segments;
    memcpy(&n_segments, input.data() + off, sizeof(n_segments));
    off += sizeof(uint32_t);

    auto it = graph_compute_infos.find(graph_number);
    if (it == graph_compute_infos.end()) {
        GGML_LOG_ERROR("[%s] graph number %d not found\n", __func__, graph_number);
        return false;
    }
    graph_info * info      = it->second->head;
    int          n_patched = 0;
    for (uint32_t s = 0; s < n_segments; s++) {
        if (off + sizeof(uint32_t) > input.size()) {
            return false;
        }
        uint32_t n_patches;
        memcpy(&n_patches, input.data() + off, sizeof(n_patches));
        off += sizeof(uint32_t);
        if (off + (size_t) n_patches * sizeof(rpc_view_patch) > input.size()) {
            return false;
        }
        const rpc_view_patch * patches = (const rpc_view_patch *) (input.data() + off);
        off += (size_t) n_patches * sizeof(rpc_view_patch);

        // (#3 prefetch) build the stride table from this real patch so a later GRAPH_ADVANCE can
        // reproduce the next token with no payload. DEFAULT ON (opt out RPC_NO_PREFETCH) = matches
        // the client gate; RPC_NO_OPT or RPC_NO_PREFETCH = plain patch apply.
        static const bool srv_prefetch = rpc_opt_enabled() && getenv("RPC_NO_PREFETCH") == nullptr;
        if (srv_prefetch && info) {  // (re)build the stride table for this segment from this patch
            info->strides.assign(info->by_idx.size(), tensor_stride{});
            info->last_patched.clear();
        }
        for (uint32_t p = 0; p < n_patches; p++) {
            const rpc_view_patch & patch = patches[p];
            if (info && patch.idx < info->by_idx.size() && info->by_idx[patch.idx] != nullptr) {
                ggml_tensor * t = info->by_idx[patch.idx];
                if (srv_prefetch) {
                    // stride = this patch's value - the currently stored value
                    tensor_stride & st = info->strides[patch.idx];
                    for (uint32_t d = 0; d < GGML_MAX_DIMS; d++) {
                        st.ne[d] = (int64_t) patch.t.ne[d] - (int64_t) t->ne[d];
                        st.nb[d] = (int64_t) patch.t.nb[d] - (int64_t) t->nb[d];
                    }
                    for (size_t j = 0; j < GGML_MAX_OP_PARAMS / sizeof(int32_t); j++) {
                        st.op_params[j] = (int64_t) patch.t.op_params[j] - (int64_t) t->op_params[j];
                    }
                    st.flags     = (int64_t) patch.t.flags - (int64_t) t->flags;
                    st.data      = (int64_t) patch.t.data - (int64_t) reinterpret_cast<uint64_t>(t->data);
                    st.view_offs = (int64_t) patch.t.view_offs - (int64_t) t->view_offs;
                    st.valid     = true;
                    info->last_patched.push_back(patch.idx);
                }
                // re-apply exactly the value-fields deserialize_tensor sets, keeping the
                // stored wiring (src/view_src/buffer) untouched. op/type don't change on a
                // hit (they're in the topology hash) so they're left as stored.
                for (uint32_t d = 0; d < GGML_MAX_DIMS; d++) {
                    t->ne[d] = patch.t.ne[d];
                    t->nb[d] = patch.t.nb[d];
                }
                memcpy(t->op_params, patch.t.op_params, sizeof(t->op_params));
                t->flags     = patch.t.flags;
                t->data      = reinterpret_cast<void *>(patch.t.data);
                t->view_offs = patch.t.view_offs;
                n_patched++;
            }
        }
        if (info) {
            info = info->next;
        }
    }
    static const bool dbg_diffcache = getenv("RPC_DBG_DIFFCACHE") != nullptr;
    if (dbg_diffcache) {
        GGML_LOG_INFO("[rpc-diffcache-srv] gnum=%d patched %d tensors over %u segments\n", graph_number, n_patched,
                      n_segments);
    }
    return true;
}

// (#3 prefetch=eliminate) advance the last-patched tensors of an already-stored graph by their
// cached per-tensor stride -- reproduces the next token's patch with NO payload bytes on the
// wire. Safe because the client sends this ONLY after verifying its predicted patch (current +
// the same stride) equals the real patch it built this token.
bool rpc_server::graph_advance(uint8_t graph_number) {
    auto it = graph_compute_infos.find(graph_number);
    if (it == graph_compute_infos.end()) {
        GGML_LOG_ERROR("[%s] graph number %d not found\n", __func__, graph_number);
        return false;
    }
    int n_adv = 0;
    for (graph_info * info = it->second->head; info; info = info->next) {
        for (uint32_t idx : info->last_patched) {
            if (idx >= info->by_idx.size() || info->by_idx[idx] == nullptr) {
                continue;
            }
            if (idx >= info->strides.size() || !info->strides[idx].valid) {
                continue;
            }
            ggml_tensor *         t  = info->by_idx[idx];
            const tensor_stride & st = info->strides[idx];
            for (uint32_t d = 0; d < GGML_MAX_DIMS; d++) {
                t->ne[d] = (int64_t) t->ne[d] + st.ne[d];
                t->nb[d] = (size_t) ((int64_t) t->nb[d] + st.nb[d]);
            }
            for (size_t j = 0; j < GGML_MAX_OP_PARAMS / sizeof(int32_t); j++) {
                t->op_params[j] = (int32_t) ((int64_t) t->op_params[j] + st.op_params[j]);
            }
            t->flags = (int32_t) ((int64_t) t->flags + st.flags);
            t->data  = reinterpret_cast<void *>((uint64_t) ((int64_t) reinterpret_cast<uint64_t>(t->data) + st.data));
            t->view_offs = (size_t) ((int64_t) t->view_offs + st.view_offs);
            n_adv++;
        }
    }
    static const bool dbg_diffcache = getenv("RPC_DBG_DIFFCACHE") != nullptr;
    if (dbg_diffcache) {
        GGML_LOG_INFO("[rpc-diffcache-srv] gnum=%d ADVANCED %d tensors (no payload)\n", graph_number, n_adv);
    }
    return true;
}

// RPC_DBG_TIMING (server side): split the do_computation time into local graph
// execute vs the all-reduce (broadcast partial + wait). Cumulative; logged per call.
static std::atomic<long long> g_srv_exec_ns{ 0 };
static std::atomic<long long> g_srv_allreduce_ns{ 0 };

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
    static const bool dbg_srv = getenv("RPC_DBG_DIFFCACHE") != nullptr;  // gate per-token lifecycle spam
    if (dbg_srv) {
        GGML_LOG_INFO("found graph number %d, doing computation\n", graph_number);
    }

    const bool opt = rpc_opt_enabled();

    // 1B (optimized only): resolve the peer sockets ONCE per token (they're kept alive in
    // peer_socks_held), instead of re-walking the map + locking weak_ptrs inside each of the
    // ~44 reduces. The baseline path re-resolves per reduce below. Re-dial an expired one.
    std::vector<std::shared_ptr<socket_t>> peer_socks;
    std::vector<std::string>               peer_names;
    if (opt) {
        for (auto & sock_weak : sockets_connectto) {
            auto sock = sock_weak.second.lock();
            if (!sock) {
                std::string host;
                int         port;
                std::string endpoint = sock_weak.first;
                if (!parse_endpoint(endpoint, host, port)) {
                    GGML_LOG_INFO("unable to parse endpoint %s", endpoint.c_str());
                }
                sock                        = socket_connect(host.c_str(), port);
                sockets_connectto[endpoint] = sock;  // existing key -> no rehash, safe mid-iteration
            }
            if (sock) {
                peer_socks.push_back(sock);
                peer_names.push_back(sock_weak.first);
            }
        }
    }

    // 1C: reuse one partial buffer across the ~44 reduces (grows as needed; no per-reduce
    // allocation -- every byte is overwritten before it's sent).
    std::vector<uint8_t> add_data;
    std::vector<float>   ar_fp32_tmp;  // reused f32 scratch for the fp16 partial conversion
    const rpc_ar_fmt     ar_fmt_sel = rpc_ar_partial();

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
            auto        _te    = std::chrono::steady_clock::now();
            ggml_status status = ggml_backend_graph_compute(backend, graph);
            GGML_ASSERT(status == GGML_STATUS_SUCCESS);
            g_srv_exec_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _te).count();
        } catch (const std::exception & e) {
            GGML_LOG_INFO("[%s] exception during graph compute: %s\n", __func__, e.what());
            ggml_free(ctx);
            return false;
        }

        // Only signal==0 segments all-reduce (1=concat, 2=no-op) -- build the partial and
        // broadcast only then (no point getting an 8KB partial for a no-op segment).
        if (signal == 0) {
            auto _ta = std::chrono::steady_clock::now();

            // build this server's partial into the reused buffer: seq(4) | src_id(4) |
            // name(GGML_MAX_NAME) | data. seq tags the token (a peer running ahead is buffered
            // by seq, not misapplied); src_id lets the receiver fold partials in a fixed order.
            const size_t     ar_hdr  = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(tensor_to_all_reduce->name);
            const size_t     nbytes  = ggml_nbytes(tensor_to_all_reduce);
            const int64_t    nelem   = ggml_nelements(tensor_to_all_reduce);
            // narrow only f32 partials (the row-split outputs are f32); else ship raw bytes.
            const rpc_ar_fmt fmt     = (tensor_to_all_reduce->type == GGML_TYPE_F32) ? ar_fmt_sel : rpc_ar_fmt::f32;
            const size_t     payload = rpc_ar_payload_bytes(fmt, nelem, nbytes);
            add_data.resize(ar_hdr + payload);                // reused; every byte set below
            uint32_t src_id = device_id;
            uint32_t seq    = ++all_reduce_seq[tensor_name];  // Nth reduce of this name == token N
            memcpy(add_data.data(), &seq, sizeof(uint32_t));
            memcpy(add_data.data() + sizeof(uint32_t), &src_id, sizeof(uint32_t));
            memcpy(add_data.data() + 2 * sizeof(uint32_t), tensor_to_all_reduce->name,
                   sizeof(tensor_to_all_reduce->name));
            if (fmt == rpc_ar_fmt::f16) {
                ar_fp32_tmp.resize((size_t) nelem);
                ggml_backend_tensor_get(tensor_to_all_reduce, ar_fp32_tmp.data(), 0, nbytes);
                ggml_fp32_to_fp16_row(ar_fp32_tmp.data(), (ggml_fp16_t *) (add_data.data() + ar_hdr), nelem);
            } else if (fmt == rpc_ar_fmt::e4m3) {  // ship fp8 e4m3[n] (1 byte/elem, no scale)
                ar_fp32_tmp.resize((size_t) nelem);
                ggml_backend_tensor_get(tensor_to_all_reduce, ar_fp32_tmp.data(), 0, nbytes);
                rpc_e4m3_quantize(ar_fp32_tmp.data(), (uint8_t *) (add_data.data() + ar_hdr), nelem);
            } else if (fmt == rpc_ar_fmt::i8b) {  // ship per-block int8 (f16 scale + int8 / 32-elem block)
                ar_fp32_tmp.resize((size_t) nelem);
                ggml_backend_tensor_get(tensor_to_all_reduce, ar_fp32_tmp.data(), 0, nbytes);
                rpc_i8b_quantize(ar_fp32_tmp.data(), (uint8_t *) (add_data.data() + ar_hdr), nelem);
            } else if (fmt == rpc_ar_fmt::i8) {  // ship scale(f32) | int8[n]
                ar_fp32_tmp.resize((size_t) nelem);
                ggml_backend_tensor_get(tensor_to_all_reduce, ar_fp32_tmp.data(), 0, nbytes);
                const float scale =
                    rpc_i8_quantize(ar_fp32_tmp.data(), (int8_t *) (add_data.data() + ar_hdr + sizeof(float)), nelem);
                memcpy(add_data.data() + ar_hdr, &scale, sizeof(float));
            } else {
                ggml_backend_tensor_get(tensor_to_all_reduce, add_data.data() + ar_hdr, 0, nbytes);
            }
            static const bool dbg_ar = getenv("RPC_DBG_AR") != nullptr;
            if (dbg_ar) {
                GGML_LOG_INFO("[ar-send] gnum=%d %s seq=%u\n", graph_number, tensor_name.c_str(), seq);
            }

            // TREE all-reduce (opt-in): non-root servers (device_id != 0) send their partial to
            // the root and AWAIT its folded result; the root receives + folds + broadcasts it.
            const bool tree    = rpc_ar_tree();
            const bool is_root = (device_id == 0);

            //set up this server's own block for this token's reduce (default-ctor + block_init
            //unifies create and re-init; a non-root in tree mode inits in await-result mode).
            block_mutex.lock();
            auto               bit = all_reduce_blocks.find(tensor_name);
            all_reduce_block * blk = (bit != all_reduce_blocks.end()) ? bit->second : nullptr;
            if (!blk) {
                blk                            = new all_reduce_block();
                all_reduce_blocks[tensor_name] = blk;
            }
            blk->block_init(tensor_to_all_reduce, signal, device_count, backend, device_id, seq, tree && !is_root);
            block_mutex.unlock();

            if (tree && is_root) {
                // ROOT: receive partials from non-roots (no broadcast here), fold, then send the
                // single f32 result to the non-roots. Result is f32 (exact) even under fp16
                // partials, so every non-root ends with the root's identical result.
                if (!blk->wait_for_completion()) {
                    GGML_LOG_ERROR("[%s] tree all-reduce TIMEOUT (root) for %s -- aborting graph\n", __func__,
                                   tensor_name.c_str());
                    return false;
                }
                const size_t         res_hdr = sizeof(uint32_t) + sizeof(tensor_to_all_reduce->name);
                std::vector<uint8_t> res(res_hdr + nbytes);  // seq(4) | name | f32 result
                memcpy(res.data(), &seq, sizeof(uint32_t));
                memcpy(res.data() + sizeof(uint32_t), tensor_to_all_reduce->name, sizeof(tensor_to_all_reduce->name));
                ggml_backend_tensor_get(tensor_to_all_reduce, res.data() + res_hdr, 0, nbytes);
                std::vector<std::thread> bcast;
                for (size_t k = 0; k < peer_socks.size(); ++k) {
                    bcast.emplace_back([&, k]() {
                        if (!send_rpc_cmd_oneway(peer_socks[k], RPC_CMD_AR_RESULT, res.data(), res.size())) {
                            GGML_LOG_INFO("failed to send ar_result to %s\n", peer_names[k].c_str());
                        }
                    });
                }
                for (auto & t : bcast) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
            } else if (tree) {
                // NON-ROOT: send our partial to the ROOT (device 0) only, then await its result.
                std::shared_ptr<socket_t> root_sock;
                if (!peer_endpoints.empty()) {
                    auto rit = sockets_connectto.find(peer_endpoints[0]);
                    if (rit != sockets_connectto.end()) {
                        root_sock = rit->second.lock();
                    }
                }
                if (!root_sock) {
                    GGML_LOG_ERROR("[%s] tree all-reduce: no socket to root for %s -- aborting graph\n", __func__,
                                   tensor_name.c_str());
                    return false;
                }
                if (!send_rpc_cmd_oneway(root_sock, RPC_CMD_ALL_REDUCE, add_data.data(), add_data.size())) {
                    GGML_LOG_INFO("failed to send partial to root for %s\n", tensor_name.c_str());
                }
                if (!blk->wait_for_completion()) {
                    GGML_LOG_ERROR("[%s] tree all-reduce TIMEOUT (non-root) for %s -- aborting graph\n", __func__,
                                   tensor_name.c_str());
                    return false;
                }
            } else if (opt) {
                // OPTIMIZED all-to-all: CONCURRENT + FIRE-AND-FORGET broadcast to all peers (one
                // thread per peer, sockets resolved once above, no ack). Each node folds locally.
                std::vector<std::thread> bcast;
                for (size_t k = 0; k < peer_socks.size(); ++k) {
                    bcast.emplace_back([&, k]() {
                        if (!send_rpc_cmd_oneway(peer_socks[k], RPC_CMD_ALL_REDUCE, add_data.data(), add_data.size())) {
                            GGML_LOG_INFO("failed to send all_reduce command to %s\n", peer_names[k].c_str());
                        }
                    });
                }
                for (auto & t : bcast) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
                if (!blk->wait_for_completion()) {
                    GGML_LOG_ERROR("[%s] all-reduce TIMEOUT for tensor %s (lost partial?) -- aborting graph\n",
                                   __func__, tensor_name.c_str());
                    return false;
                }
            } else {
                // BASELINE: SERIAL + ACKED broadcast to all peers (the pre-optimization path;
                // the matching server ALL_REDUCE handler replies in baseline mode too).
                for (auto & sock_weak : sockets_connectto) {
                    auto sock = sock_weak.second.lock();
                    if (!sock) {
                        std::string host;
                        int         port;
                        std::string endpoint = sock_weak.first;
                        if (!parse_endpoint(endpoint, host, port)) {
                            GGML_LOG_INFO("unable to parse endpoint %s", endpoint.c_str());
                        }
                        sock                        = socket_connect(host.c_str(), port);
                        sockets_connectto[endpoint] = sock;
                    }
                    if (sock && !send_rpc_cmd(sock, RPC_CMD_ALL_REDUCE, add_data.data(), add_data.size(), nullptr, 0)) {
                        GGML_LOG_INFO("failed to send all_reduce command to %s\n", sock_weak.first.c_str());
                    }
                }
                if (!blk->wait_for_completion()) {
                    GGML_LOG_ERROR("[%s] all-reduce TIMEOUT for tensor %s (lost partial?) -- aborting graph\n",
                                   __func__, tensor_name.c_str());
                    return false;
                }
            }
            // Reset UNDER block_mutex: block_uinit() frees add_tensor->buffer + ctx (where the
            // add graph lives), and a peer thread runs add() under block_mutex -- without this
            // lock the free could race a peer still inside add() => use-after-free. This is THE
            // intermittent all-reduce crash (Heisenbug: extra logging slowed the race away).
            block_mutex.lock();
            blk->block_uinit();
            block_mutex.unlock();
            g_srv_allreduce_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _ta).count();
        }
    }

    static const bool dbg_timing = getenv("RPC_DBG_TIMING") != nullptr;
    if (dbg_timing) {
        long long ex = g_srv_exec_ns.load();
        long long ar = g_srv_allreduce_ns.load();
        GGML_LOG_INFO("[rpc-srv-timing] cumulative exec=%.2fs all_reduce=%.2fs (exec=%.1f%% all_reduce=%.1f%%)\n",
                      ex / 1e9, ar / 1e9, 100.0 * ex / (double) (ex + ar + 1), 100.0 * ar / (double) (ex + ar + 1));
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
    // device id -> endpoint map (lets the tree all-reduce find the root, device 0)
    peer_endpoints.assign(request.device_count, std::string());
    for (uint8_t i = 0; i < request.device_count; i++) {
        peer_endpoints[i] = request.endpoints[i];
    }
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
        all_reduce_seq.clear();  // restart all-reduce sequence numbers for the new run (both peers reset together)
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

    //parse seq | src_id | tensor_name | tensor_data (layout matches do_computation's add_data)
    uint32_t seq;
    memcpy(&seq, input.data(), sizeof(uint32_t));
    uint32_t src_id;
    memcpy(&src_id, input.data() + sizeof(uint32_t), sizeof(uint32_t));
    char tensor_name_[GGML_MAX_NAME];
    memcpy(tensor_name_, input.data() + 2 * sizeof(uint32_t), sizeof(tensor_name_));
    std::vector<uint8_t> tensor_data;
    size_t               data_off = 2 * sizeof(uint32_t) + sizeof(tensor_name_);
    tensor_data.resize(input.size() - data_off, 0);
    memcpy(tensor_data.data(), input.data() + data_off, tensor_data.size());

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
    auto              it     = all_reduce_blocks.find(tensor_name);
    static const bool dbg_ar = getenv("RPC_DBG_AR") != nullptr;
    if (dbg_ar) {
        GGML_LOG_INFO("[ar-recv] %s seq=%u cur=%u exists=%d init=%d\n", tensor_name.c_str(), seq,
                      it == all_reduce_blocks.end() ? 0 : it->second->get_current_seq(), it != all_reduce_blocks.end(),
                      it != all_reduce_blocks.end() && it->second->is_init());
    }
    if (it == all_reduce_blocks.end()) {
        // no block yet -> buffer this partial under its sequence until we reach it
        try {
            all_reduce_block * block = new all_reduce_block();
            block->add_to_buffer(seq, (uint8_t) src_id, tensor_data);
            all_reduce_blocks[tensor_name] = block;
        } catch (const std::exception & e) {
            GGML_LOG_INFO("[%s] error: %s\n", __func__, e.what());
        }
    } else if (it->second->is_init() && seq == it->second->get_current_seq()) {
        //the block is reducing exactly this token -> apply now
        it->second->add(tensor_data, (uint8_t) src_id);
    } else if (seq > it->second->get_current_seq()) {
        //peer is ahead of us on this tensor -> buffer until we reach this token
        it->second->add_to_buffer(seq, (uint8_t) src_id, tensor_data);
    }
    //else: seq <= current_seq but not the active reduce -> stale duplicate, drop
    block_mutex.unlock();
    return true;
}

// TREE all-reduce: a NON-ROOT received the root's folded result. Layout: seq(4) | name | data
// (f32, no src_id -- it's the single result). Mirrors all_reduce()'s seq matching/buffering.
bool rpc_server::ar_result(std::vector<uint8_t> & input) {
    uint32_t seq;
    memcpy(&seq, input.data(), sizeof(uint32_t));
    char tensor_name_[GGML_MAX_NAME];
    memcpy(tensor_name_, input.data() + sizeof(uint32_t), sizeof(tensor_name_));
    size_t               data_off = sizeof(uint32_t) + sizeof(tensor_name_);
    std::vector<uint8_t> result(input.begin() + data_off, input.end());
    std::string          tensor_name = tensor_name_;

    block_mutex.lock();
    auto it = all_reduce_blocks.find(tensor_name);
    if (it == all_reduce_blocks.end()) {
        // result arrived before we set up this token's block -> buffer it by seq
        try {
            all_reduce_block * block = new all_reduce_block();
            block->add_result_to_buffer(seq, result);
            all_reduce_blocks[tensor_name] = block;
        } catch (const std::exception & e) {
            GGML_LOG_INFO("[%s] error: %s\n", __func__, e.what());
        }
    } else if (it->second->is_init() && seq == it->second->get_current_seq()) {
        it->second->set_result(result);
    } else if (seq > it->second->get_current_seq()) {
        it->second->add_result_to_buffer(seq, result);
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
            case RPC_CMD_SEND_TO_PEER:
                {
                    rpc_msg_send_to_peer_req request;
                    if (!recv_msg(sockfd, &request, sizeof(request))) {
                        return;
                    }
                    rpc_msg_send_to_peer_rsp response;
                    if (!server.send_to_peer(request, response)) {
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
            case RPC_CMD_GRAPH_COMPUTE_BATCH:
                {
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    rpc_msg_graph_compute_rsp response;
                    if (!server.graph_compute_batch(input, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_GRAPH_COMPUTE_STORE:  // (PP diff cache) non-split: deserialize + store + compute inline
                {
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    rpc_msg_graph_compute_rsp response;
                    if (!server.graph_compute_store(input, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_PATCH_COMPUTE:  // (PP diff cache) non-split: patch stored graph + compute inline
                {
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    rpc_msg_graph_compute_rsp response;
                    if (!server.patch_compute(input, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_ADVANCE_COMPUTE:  // (PP diff cache + prefetch) non-split: advance by stride + compute inline
                {
                    uint8_t graph_number;
                    if (!recv_msg(sockfd, &graph_number, sizeof(graph_number))) {
                        return;
                    }
                    rpc_msg_graph_compute_rsp response;
                    if (!server.advance_compute(graph_number, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_PATCH_VIEWS:
                {
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    if (!server.patch_views(input)) {
                        return;
                    }
                    // skip the ack when the client sent this one-way (else the ack bytes pollute
                    // the next DO_COMPUTATION response on this fd).
                    if (!rpc_graph_oneway() && !send_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_GRAPH_ADVANCE:
                {
                    uint8_t graph_number;
                    if (!recv_msg(sockfd, &graph_number, sizeof(graph_number))) {
                        return;
                    }
                    if (!server.graph_advance(graph_number)) {
                        return;
                    }
                    if (!rpc_graph_oneway() && !send_msg(sockfd, nullptr, 0)) {
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
                    // Optimized: fire-and-forget -- the peer used send_rpc_cmd_oneway and is NOT
                    // waiting for a reply, so don't send one (an unread ack would pile up in the
                    // peer's recv buffer). Baseline (RPC_NO_OPT): the peer used the acked
                    // send_rpc_cmd, so we MUST reply. Both sides read the same gate, so they
                    // agree -- the env var must be set consistently across all rpc-servers.
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    if (!server.all_reduce(input)) {
                        return;
                    }
                    if (!rpc_opt_enabled() && !send_msg(sockfd, nullptr, 0)) {
                        return;
                    }
                    break;
                }
            case RPC_CMD_AR_RESULT:
                {
                    // tree all-reduce: root -> non-root result. Fire-and-forget (sent via
                    // send_rpc_cmd_oneway), so no reply.
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    if (!server.ar_result(input)) {
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
            case RPC_CMD_BATCH_LOAD_CACHED:
                {
                    std::vector<uint8_t> input;
                    if (!recv_msg(sockfd, input)) {
                        return;
                    }
                    std::vector<uint8_t> response;
                    if (!server.batch_load_cached(input, response)) {
                        return;
                    }
                    if (!send_msg(sockfd, response.data(), response.size())) {
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
