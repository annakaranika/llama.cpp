"""Run TTFT and TPT evaluation across policies."""

from graph_partitioning import Device, ModelSpec, Network, Simulator
from graph_partitioning.common import CommunicationModel, PeerToPeerPolicy

def main():
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
    simulator = Simulator(net, model, output_dir="sim_output")
    simulator.evaluate_ttft_tpt(prompt_lengths=[128, 256, 512])

if __name__ == "__main__":
    main()
