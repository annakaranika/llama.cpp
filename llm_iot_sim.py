"""
llm_iot_sim.py
----------------
A lightweight simulator for distributing LLM inference across heterogeneous (wireless) IoT devices.

Supports policies:
  - "single_node": everything on one device.
  - "pipeline": layer-wise pipeline parallelism across devices.
  - "tensor": tensor parallelism across a group of devices with ring all-reduce model.
  - "metis": METIS k-way partitioning of the layer graph (if PyMetis is available),
    else falls back to a greedy balancer.
  - "alpa": Alpa-style 2D partitioning (pipeline + tensor) with attention splitting.
  - "sct": SCT (Smallest Communication Time) partitioning.

Model notes (TinyLlama-like by default):
  - 23 layers (including output head), 32 Q heads, 4 KV heads.
  - You can adjust model dimensions and coefficients for FLOPs and activation sizes to
    fit your needs.

Network model:
  - Simple symmetric wireless links (same RTT/bandwidth for all pairs) by default.
  - You can override a pairwise matrix if needed.
"""

import datetime
from graph_partitioning import Device, Network, ModelSpec, Simulator
from graph_partitioning.common import CommunicationModel, PeerToPeerPolicy


if __name__ == "__main__":
    # Example: 4 heterogeneous Raspberry Pi-like devices
    CLIENT = Device("client", gflops=100.0, memory_gb=16.0)
    SERVERS = [
        Device("rpi-1", gflops=1.0, memory_gb=2.0),
        Device("rpi-2", gflops=3.0, memory_gb=1.0),
        Device("rpi-3", gflops=5.5, memory_gb=2.0),
        Device("rpi-4", gflops=0.4, memory_gb=1.0),
        # Device("rpi-5", gflops=8.0, memory_gb=1.0),
        # Device("rpi-6", gflops=6.0, memory_gb=2.0),
        # Device("rpi-7", gflops=4.5, memory_gb=2.0),
        # Device("rpi-8", gflops=0.5, memory_gb=4.0),
    ]

    # Shared wireless fabric
    NET = Network(CLIENT, SERVERS, default_bw_mbps=200.0, default_rtt_ms=18.0)
    NET.set_pairwise_links(filename="pair_links.csv")
    NET.set_comm_config(
        communication_model=CommunicationModel.PEER_TO_PEER,
        peer2peer_policy=PeerToPeerPolicy.ALL_TO_ALL,
    )

    # Model + workload
    MODEL = ModelSpec()
    SEQ_LEN = 1
    GEN_TOKENS = 1

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"sim_output/{timestamp}"
    simulator = Simulator(NET, MODEL, OUTPUT_DIR)

    res = simulator.compare_sched_policies(SEQ_LEN, GEN_TOKENS)
    sorted_res = sorted(res.items(), key=lambda x: x[1])
    print("Final results:")
    for k, v in sorted_res:
        print(f"  {k:<30}: {v:>8.2f} s")
