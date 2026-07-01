"""Network model and communication cost estimation for distributed LLM inference."""

import csv
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .common import (
    CommunicationModel,
    DevNetAggregationStrategy,
    Device,
    PeerToPeerPolicy,
)


def _load_links_from_csv(filename: str) -> List[Tuple[str, str, float, float]]:
    """
    Load pairwise links from a CSV file.
    Expected format: device1,device2,bw_mbps,rtt_ms
    """
    links = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip header if present
        header = next(reader, None)
        if header and header[2].replace(".", "").isdigit():
            # First row is data, add it
            device_a, device_b, bw_str, rtt_str = header
            links.append((device_a, device_b, float(bw_str), float(rtt_str)))

        for row in reader:
            if len(row) >= 4:
                device_a, device_b, bw_str, rtt_str = row[:4]
                links.append((device_a, device_b, float(bw_str), float(rtt_str)))
    return links


@dataclass
class Network:
    """
    Represents a network with symmetric shared wireless fabric.
    If pairwise overrides are provided, they take precedence when computing link costs.
    """

    coordinator: Optional[Device] = None
    servers: List[Device] = field(default_factory=list)
    # ``coordinator`` is the device running the inference driver (llama-bench/
    # llama-cli): it injects the input to the first stage and collects the output
    # from the last. It may be an OFF-cluster driver (a Mac over the AP, the
    # default) OR one of the ``servers`` (an on-cluster RPi), in which case
    # transfers to/from a stage it co-hosts are free. ``coordinator_name`` is kept
    # in sync with ``coordinator.name`` by __post_init__; set just the name (to a
    # server's name) to pick an on-cluster coordinator and the device is resolved.
    coordinator_name: Optional[str] = None
    communication_model: CommunicationModel = CommunicationModel.CLIENT_SERVER
    peer2peer_policy: Optional[PeerToPeerPolicy] = None
    dev_net_aggregation_strategy: Optional[DevNetAggregationStrategy] = (
        DevNetAggregationStrategy.SUM
    )

    default_bw_mbps: float = 300.0
    default_rtt_ms: float = 15.0
    pair_bw_mbps: Dict[Tuple[str, str], float] = field(default_factory=dict)
    pair_rtt_ms: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Communication model parameters for centralized vs peer-to-peer
    client_bandwidth_mbps: float = 1000.0  # Client-server bandwidth
    client_rtt_ms: float = 10.0  # Client-server RTT

    compute_coeff: float = 1.0
    mem_coeff: float = 1.0
    rtt_coeff: float = 1.0

    # Fixed per-transfer overhead (s) for llama.cpp's RPC handshake/serialization,
    # added to every non-self transfer. The raw bw+rtt model misses the per-message
    # cost of the RPC backend, so the sim under-predicts whenever a hop is RPC-routed
    # (e.g. coordinator -> backend). Calibrate from a 1-RPC-hop scenario; default 0
    # reproduces the original behavior.
    msg_overhead_s: float = 0.0
    # RPC bandwidth efficiency η ∈ (0, 1]: the fraction of the measured (iperf) link
    # bandwidth that llama.cpp's RPC backend actually achieves for an activation
    # transfer. Synchronous protocol, per-tensor serialization, copies and the lack
    # of compute/transfer overlap mean the effective bandwidth is well below the raw
    # link — so the byte term is divided by η. Applies to EVERY transfer (inter-stage
    # hops and coordinator I/O both route through xfer_time_s). Default 1.0 = the
    # original raw-bandwidth behavior; calibrate from the prefill-vs-N slope.
    rpc_efficiency: float = 1.0
    # When True, server<->server transfers relayed through a CLIENT coordinator share
    # that one AP uplink: the effective bandwidth is divided across the backends,
    # modelling AP contention as the cluster grows. Only affects client-coordinated
    # (Mac/AP) relays; on-mesh RPi coordinators are unaffected. Default False =
    # original behavior. (Does NOT reproduce the pathological n=16 saturation cliff.)
    hub_bw_share: bool = False

    def set_comm_config(
        self,
        communication_model: CommunicationModel,
        peer2peer_policy: Optional[PeerToPeerPolicy] = None,
    ):
        """
        Set the communication configuration for the network.
        """
        self.communication_model = communication_model
        if communication_model == CommunicationModel.CLIENT_SERVER:
            self.peer2peer_policy = None
            return
        if self.peer2peer_policy is None and peer2peer_policy is None:
            raise ValueError(
                "peer2peer_policy must be specified for peer-to-peer model"
            )
        self.peer2peer_policy = peer2peer_policy

    def set_pairwise_links(
        self,
        links: Optional[List[Tuple[str, str, float, float]]] = None,
        filename: Optional[str] = None,
    ):
        """
        Set pairwise bandwidth (Mbps) and RTT (ms) overrides between devices.
        Each link is a tuple of (device_a, device_b, bandwidth_mbps, rtt_ms).
        """
        if links is None and filename is None:
            raise ValueError("Either links or filename must be provided")

        if filename:
            links = _load_links_from_csv(filename)

        assert links is not None
        for a, b, bw, rtt in links:
            self.pair_bw_mbps[(a, b)] = bw
            self.pair_rtt_ms[(a, b)] = rtt

    def set_dev_net_aggregation_strategy(self, dev_net_aggr: DevNetAggregationStrategy):
        """
        Set the device network aggregation strategy for bandwidth estimation.
        """
        self.dev_net_aggregation_strategy = dev_net_aggr

    def __post_init__(self):
        # Resolve the coordinator device and keep ``coordinator``/``coordinator_name``
        # in sync. If only a name is given, use the matching server (on-cluster) or
        # a synthesised off-cluster driver with that name (its compute/memory are
        # unused — only the name matters, for link routing).
        if self.coordinator is None:
            if self.coordinator_name is not None:
                self.coordinator = next(
                    (s for s in self.servers if s.name == self.coordinator_name),
                    Device(self.coordinator_name, gflops=15.0, memory_gb=4.0),
                )
            else:
                self.coordinator = Device("coordinator", gflops=15.0, memory_gb=4.0)
        if self.coordinator_name is None:
            self.coordinator_name = self.coordinator.name

    def link(self, a: str, b: str) -> Tuple[float, float]:
        """
        Get bandwidth (Mbps) and RTT (ms) between two devices.

        In CLIENT_SERVER mode every server-to-server transfer is relayed by the
        coordinator (llama.cpp's RPC backend ferries all tensors), so a<->b is
        modelled as a->coordinator->b: bw = min of the two legs, rtt = their sum
        (store-and-forward). Relaying through the ``coordinator`` — which may be an
        off-cluster driver (a Mac over the AP) or an on-cluster RPi — is what makes
        a Mac coordinator pay an AP crossing on every leg while an RPi coordinator
        keeps the traffic in the mesh. The ``a != hub and b != hub`` guard both
        avoids infinite recursion and returns a direct link for the leg that
        terminates at the coordinator.
        """
        if a == b:
            return float("inf"), 0.0
        key = (a, b)

        hub = self.coordinator_name
        if (
            self.communication_model == CommunicationModel.CLIENT_SERVER
            and a != hub
            and b != hub
        ):
            bw_a, rtt_a = self.link(a, hub)
            bw_b, rtt_b = self.link(hub, b)
            bw = min(bw_a, bw_b)
            off_cluster = not any(s.name == hub for s in self.servers)
            if self.hub_bw_share and off_cluster and self.servers:
                # backends contend for the one AP uplink at an off-cluster coordinator
                bw /= len(self.servers)
            return bw, rtt_a + rtt_b

        if key in self.pair_bw_mbps:
            bw = self.pair_bw_mbps[key]
        elif (b, a) in self.pair_bw_mbps:
            bw = self.pair_bw_mbps[(b, a)]
        else:
            bw = self.default_bw_mbps

        if key in self.pair_rtt_ms:
            rtt = self.pair_rtt_ms[key]
        elif (b, a) in self.pair_rtt_ms:
            rtt = self.pair_rtt_ms[(b, a)]
        else:
            rtt = self.default_rtt_ms
        return bw, rtt

    def xfer_time_s(self, bytes_size: float, a: str, b: str) -> float:
        """
        Estimate transfer time between two devices considering bandwidth and RTT.
        """
        bw_mbps, rtt_ms = self.link(a, b)
        if math.isinf(bw_mbps):
            return 0.0  # self-link (co-located): free, no RPC message either
        return (
            bytes_size * 8.0 / (bw_mbps * 1e6 * self.rpc_efficiency)
            + rtt_ms / 2 / 1000.0
            + self.msg_overhead_s  # per-message RPC handshake (0 by default)
        )

    def coordinator_io_time_s(
        self,
        in_bytes: float,
        out_bytes: float,
        entry_dev: str,
        exit_dev: str,
    ) -> float:
        """Per-forward coordinator I/O: inject the input to the entry device and
        read the output (logits) back from the exit device.

        CLIENT_SERVER only — that is the regime with a driver process in the
        loop (llama.cpp's RPC backend). In PEER_TO_PEER re is no coordinator,
        so this is 0. A leg that terminates at the coordinator's own device
        resolves to a self-link (also 0). Shared by every policy so the gating
        and the routing live in exactly one place.
        """
        if self.communication_model != CommunicationModel.CLIENT_SERVER:
            return 0.0
        hub = self.coordinator_name
        return (
            self.xfer_time_s(in_bytes, hub, entry_dev)
            + self.xfer_time_s(out_bytes, exit_dev, hub)
        )

    def avg_client_server_time(
        self, bytes_size: float, servers: Optional[List[Device]] = None
    ) -> float:
        """
        Client-server transfer time model.
        """
        if servers is None:
            servers = self.servers
        bw_rtt_pairs = [self.link(self.coordinator_name, server.name) for server in servers]
        if bw_rtt_pairs:
            avg_bw = sum(bw for bw, _ in bw_rtt_pairs) / len(bw_rtt_pairs)
            avg_rtt = sum(rtt for _, rtt in bw_rtt_pairs) / len(bw_rtt_pairs)
        else:
            avg_bw = self.client_bandwidth_mbps
            avg_rtt = self.client_rtt_ms

        return (bytes_size * 8.0 / (avg_bw * 1e6)) + (avg_rtt / 2 / 1000.0)

    def avg_server_server_time(
        self, bytes_size: float, servers: Optional[List[Device]] = None
    ) -> float:
        """
        Average server-server transfer time model.
        """
        if servers is None:
            servers = self.servers
        n = len(servers)
        if n <= 1:
            return 0.0

        total_bw = 0.0
        total_rtt = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                bw, rtt = self.link(servers[i].name, servers[j].name)
                total_bw += bw
                total_rtt += rtt
                count += 1

        avg_bw = total_bw / count
        avg_rtt = total_rtt / count

        return (bytes_size * 8.0 / (avg_bw * 1e6)) + (avg_rtt / 2 / 1000.0)

    def aggregate_bandwidth_capacity(self, server: Device) -> float:
        """
        Estimate the aggregate bandwidth (Mbps) for a given server.
        """
        other_servers = [s for s in self.servers if s.name != server.name]
        if self.dev_net_aggregation_strategy == DevNetAggregationStrategy.SUM:
            return sum(self.link(server.name, s.name)[0] for s in other_servers)
        if self.dev_net_aggregation_strategy == DevNetAggregationStrategy.MAX:
            return max(self.link(server.name, s.name)[0] for s in other_servers)
        if self.dev_net_aggregation_strategy == DevNetAggregationStrategy.MIN:
            return min(self.link(server.name, s.name)[0] for s in other_servers)
        raise ValueError("Unknown DevNetAggregationStrategy")

    def centralized_allreduce_time(
        self,
        tensor_size_bytes: float,
        leader: Device,
        servers: Optional[List[Device]] = None,
    ) -> float:
        """
        Centralized all-reduce communication policy.
        Assumes a leader device coordinates all communication.
        """
        if servers is None:
            servers = self.servers

        # Actual RPC sequence:
        #   Graph Compute
        #   -> num_servers x {Set Partial Tensor -> Add Data Compute Graph -> Get Partial Tensor}
        #   -> Set Final Aggregated Tensor
        add_graph_size = 512  # bytes for simple add op graph
        block_graph_size = 1024  # bytes for small compute graph
        total_cost = 0.0

        # Phase 1: Client sends graph compute request to all servers (parallel)
        max_graph_compute_time = 0.0
        for server in servers:
            # Graph compute request (serialized subgraph + tensor descriptors)
            # Estimate ~1KB for small subgraph + tensor metadata per server
            request_time = self.xfer_time_s(block_graph_size, leader.name, server.name)
            # Response is just status (small)
            response_time = self.xfer_time_s(0, server.name, leader.name)
            graph_compute_time = request_time + response_time
            max_graph_compute_time = max(max_graph_compute_time, graph_compute_time)

        total_cost += max_graph_compute_time

        # Phase 2: Client performs partial tensor summation with all servers (sequential)
        sum_set_add_get_time = 0.0
        for server in servers:
            # Set tensor request (tensor data)
            partial_set_request_time = self.xfer_time_s(
                tensor_size_bytes, leader.name, server.name
            )
            # Response is just status (small)
            partial_set_response_time = self.xfer_time_s(0, server.name, leader.name)
            # Add tensor request (small tensor descriptor + add op graph)
            add_request_time = self.xfer_time_s(
                add_graph_size, leader.name, server.name
            )
            # Response is just status (small)
            add_response_time = self.xfer_time_s(0, server.name, leader.name)
            # Get tensor request (small tensor descriptor)
            partial_get_request_time = self.xfer_time_s(0, server.name, leader.name)
            # Response contains the actual tensor data
            partial_get_response_time = self.xfer_time_s(
                tensor_size_bytes, leader.name, server.name
            )
            per_server_time = (
                partial_set_request_time
                + partial_set_response_time
                + add_request_time
                + add_response_time
                + partial_get_request_time
                + partial_get_response_time
            )
            sum_set_add_get_time += per_server_time

        total_cost += sum_set_add_get_time

        # Phase 3: Client sends aggregated tensor back to all servers
        # for local storage/reduction (parallel)
        max_final_set_time = 0.0
        for server in servers:
            # All reduce compute graph request
            # (simple addition graph for local all-reduce + full aggregated tensor)
            # bytes for simple add op graph
            request_time = self.xfer_time_s(tensor_size_bytes, leader.name, server.name)
            # Response is just computation status
            response_time = self.xfer_time_s(0, server.name, leader.name)
            final_set_time = request_time + response_time
            max_final_set_time = max(max_final_set_time, final_set_time)

        total_cost += max_final_set_time

        # NOTE: no fixed per-server "sync overhead" is added. A hardcoded
        # 0.001 * len(servers) (1 ms/server) used to live here, but it was
        # network-insensitive and dominated the cost on fast fabrics (~4 ms vs
        # ~0.6 ms of actual transfer for a 4-device all-reduce on a 100 Gbps
        # link), making the collective never worth it regardless of bandwidth.
        # The per-message RTTs charged above (every request/response, including
        # 0-byte status replies, pays rtt/2 via xfer_time_s -> link) already
        # model coordination latency, and they scale with the actual links.
        return total_cost

    def ring_allreduce_time_with_pairwise_links(
        self, bytes_size: float, servers: Optional[List[Device]] = None
    ) -> float:
        """
        Ring all-reduce time accounting for actual pairwise link characteristics.
        In a ring, each device sends to its next neighbor in the ring.
        Following this model:
            time ≈ 2 * (N - 1) / N * (bytes / bw) + (N - 1) * RTT

        TODO: take into consideration the case that splits are not the same number
        between attention and FFN blocks.
        """
        if servers is None:
            servers = self.servers
        n = len(servers)
        if n <= 1:
            return 0.0

        # Build ring: device[i] -> device[(i+1) % n]
        total_time = 0.0

        # Reduce-scatter phase: (n-1) steps, each step uses one ring link
        for _ in range(n - 1):
            # Find the bottleneck link for this step (all devices send simultaneously)
            step_time = 0.0
            for i in range(n):
                sender = servers[i].name
                receiver = servers[(i + 1) % n].name

                # Each step sends 1/n of the data
                chunk_size = bytes_size / n
                link_time = self.xfer_time_s(chunk_size, sender, receiver)
                step_time = max(step_time, link_time)  # Bottleneck determines step time

            total_time += step_time

        # All-gather phase: (n-1) more steps
        for _ in range(n - 1):
            step_time = 0.0
            for i in range(n):
                sender = servers[i].name
                receiver = servers[(i + 1) % n].name

                # Each step sends 1/n of the data
                chunk_size = bytes_size / n
                link_time = self.xfer_time_s(chunk_size, sender, receiver)
                step_time = max(step_time, link_time)

            total_time += step_time

        return total_time

    def allreduce_communication_cost(  # pylint: disable=too-many-locals
        self,
        tensor_size_bytes: float,
        servers: Optional[List[Device]] = None,
        **kwargs,
    ) -> float:
        """Compute communication cost based on the model type and actual pairwise links."""
        if servers is None:
            servers = self.servers
        if len(servers) <= 1:
            return 0.0

        communication_model = kwargs.get(
            "communication_model", self.communication_model
        )
        if communication_model == CommunicationModel.CLIENT_SERVER:
            # Centralized all-reduce using client-server links. Forward `servers`
            # so the all-reduce is priced over the ACTUAL participants (which may
            # be a sharded subset), not always the whole cluster — this preserves
            # e.g. msct's favorite-child benefit (fewer participants => cheaper).
            assert self.coordinator is not None
            return self.centralized_allreduce_time(
                tensor_size_bytes, leader=self.coordinator, servers=servers
            )

        peer2peer_policy = kwargs.get("peer2peer_policy", self.peer2peer_policy)
        if peer2peer_policy == PeerToPeerPolicy.CENTRALIZED:
            # Centralized all-reduce using leader server-server links
            return self.centralized_allreduce_time(tensor_size_bytes, leader=servers[0])

        if peer2peer_policy == PeerToPeerPolicy.ALL_TO_ALL:
            # All-to-all all-reduce using actual pairwise links
            # Simplified model: each device sends/receives full tensor to/from all ors
            total_time = 0.0
            for sender in servers:
                max_send_time = 0.0
                for receiver in servers:
                    if sender == receiver:
                        continue
                    link_time = self.xfer_time_s(
                        tensor_size_bytes, sender.name, receiver.name
                    )
                    max_send_time = max(max_send_time, link_time)
                total_time = max(
                    total_time, max_send_time
                )  # Each device sends in parallel
            return total_time

        if peer2peer_policy == PeerToPeerPolicy.HIERARCHICAL:
            raise NotImplementedError("Hierarchical all-reduce not implemented yet.")

        # Ring all-reduce using actual pairwise links
        return self.ring_allreduce_time_with_pairwise_links(tensor_size_bytes)

    def filter_servers_by_memory(
        self, required_memory_gb: float, servers: Optional[List[Device]] = None
    ) -> List[Device]:
        """
        Filter servers that have at least the required memory.
        """
        if servers is None:
            servers = self.servers
        return [server for server in servers if server.memory_gb >= required_memory_gb]

    def find_fastest_server(
        self, servers: Optional[List[Device]] = None
    ) -> Optional[Device]:
        """
        Find the server with the highest compute capability (GFLOPS).
        """
        if servers is None:
            servers = self.servers
        return max(servers, key=lambda server: server.gflops)

    def find_closest_server(
        self, curr: Device, servers: Optional[List[Device]] = None
    ) -> Optional[Device]:
        """
        Find the server with the lowest RTT from the coordinator.
        """
        if servers is None:
            servers = self.servers
        return min(servers, key=lambda server: self.link(curr.name, server.name)[1])

    def compute_device_scores(
        self, candidates: List[Device], chosen_devs: List[Device]
    ) -> List[float]:
        """
        Compute a score for all candidate devices based on ir compute power and
        average RTT to already chosen devices.
        Higher compute and lower RTT yield a higher score.
        """

        computes = []
        memories = []
        avg_rtts = []
        for candidate in candidates:
            # Compute average RTT to chosen devices
            total_rtt = 0.0
            for dev in chosen_devs:
                _, rtt = self.link(candidate.name, dev.name)
                total_rtt += rtt
            avg_rtt = total_rtt / len(chosen_devs) if chosen_devs else 1.0
            avg_rtts.append(avg_rtt)
            computes.append(candidate.gflops)
            memories.append(candidate.memory_gb)

        scores = []
        for compute, memory, avg_rtt in zip(computes, memories, avg_rtts):
            score = (
                self.compute_coeff * compute / sum(computes)
                + self.mem_coeff * memory / sum(memories)
                + self.rtt_coeff * sum(avg_rtts) / avg_rtt
            )
            scores.append(score)
        return scores
