"""Run every partitioning policy on a sample network and emit comparison plots.

Outputs a heatmap (per-layer device share), a total-latency bar chart, and a
device-workload share figure under ``sim_output/<timestamp>/compare``.
"""

import datetime

from graph_partitioning import Device, ModelSpec, Network, Simulator
from graph_partitioning.common import CommunicationModel, PeerToPeerPolicy


def main() -> None:
    coordinator = Device("coordinator", gflops=100.0, memory_gb=16.0)
    servers = [
        Device("rpi-1", gflops=1.0, memory_gb=2.0),
        Device("rpi-2", gflops=3.0, memory_gb=1.0),
        Device("rpi-3", gflops=5.5, memory_gb=0.5),
        Device("rpi-4", gflops=0.4, memory_gb=1.0),
    ]

    net = Network(coordinator, servers, default_bw_mbps=200.0, default_rtt_ms=18.0)
    net.set_pairwise_links(filename="pair_links_mesh.csv")
    net.set_comm_config(
        communication_model=CommunicationModel.PEER_TO_PEER,
        peer2peer_policy=PeerToPeerPolicy.ALL_TO_ALL,
    )

    model = ModelSpec()
    seq_len = 256
    gen_tokens = 64

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"sim_output/{timestamp}"
    simulator = Simulator(net, model, output_dir)

    simulator.plot_all_partitions(seq_len=seq_len, gen_tokens=gen_tokens)


if __name__ == "__main__":
    main()
