"""Unit tests for Alpa policy simulation in graph partitioning."""

import os
from unittest.mock import Mock, patch

import pytest

from graph_partitioning.policies.alpa import simulate_alpa, _alpa_form_clusters
from graph_partitioning.common import CommunicationModel, PeerToPeerPolicy
from graph_partitioning import Device, ModelSpec, Network


@pytest.fixture
def mock_model() -> Mock:
    """Generates a mock ModelSpec with predefined attributes and methods."""
    model = Mock(spec=ModelSpec)
    model.num_heads = 32
    model.num_kv_heads = 8
    model.num_layers = 24
    model.attn_allreduce_bytes = 1024
    model.ffn_allreduce_bytes = 2048
    model.activation_bytes = 4096
    model.layer_compute_flops.return_value = 1000000
    model.layer_compute_cost_per_device.return_value = [0.1, 0.1]
    return model


@pytest.fixture
def mock_network() -> Mock:
    """Generates a mock Network with two devices and predefined link characteristics."""
    net = Mock(spec=Network)

    # Mock devices
    device1 = Mock(spec=Device)
    device1.name = "rpi-1"
    device1.flops_per_sec.return_value = 10000000

    device2 = Mock(spec=Device)
    device2.name = "rpi-2"
    device2.flops_per_sec.return_value = 10000000

    net.servers = [device1, device2]
    net.link.return_value = (100.0, 0.01)  # bandwidth, latency
    net.ring_allreduce_time_with_pairwise_links.return_value = 0.001
    net.allreduce_communication_cost.return_value = 0.002

    return net


def network_multiple_devices() -> Network:
    """Generates a Network with 4 servers and default links."""
    device1 = Device("rpi-1", gflops=5.0, memory_gb=2.0)
    device2 = Device("rpi-2", gflops=5.0, memory_gb=2.0)
    device3 = Device("rpi-3", gflops=5.0, memory_gb=2.0)
    device4 = Device("rpi-4", gflops=5.0, memory_gb=2.0)
    device5 = Device("rpi-5", gflops=5.0, memory_gb=2.0)
    device6 = Device("rpi-6", gflops=5.0, memory_gb=2.0)

    servers = [device1, device2, device3, device4, device5, device6]
    communication_model = CommunicationModel.PEER_TO_PEER
    peer2peer_policy = PeerToPeerPolicy.ALL_TO_ALL

    net = Network(
        servers=servers,
        communication_model=communication_model,
        peer2peer_policy=peer2peer_policy,
        default_bw_mbps=50.0,
        default_rtt_ms=10.0,
    )

    return net


@pytest.fixture
def network_multiple_servers_mesh() -> Network:
    """Generates a Network with 6 servers and mesh topology links."""

    net = network_multiple_devices()
    pair_links_filename = os.path.join(
        os.path.dirname(__file__), "../pair_links_mesh.csv"
    )
    net.set_pairwise_links(filename=pair_links_filename)

    return net


@pytest.fixture
def network_multiple_devices_ap() -> Network:
    """Generates a Network with 6 servers and AP-based topology links."""

    net = network_multiple_devices()
    pair_links_filename = os.path.join(
        os.path.dirname(__file__), "../pair_links_ap.csv"
    )
    net.set_pairwise_links(filename=pair_links_filename)

    return net


def test_simulate_alpa_single_device(
    mock_model, mock_network
):  # pylint: disable=redefined-outer-name
    """Test Alpa simulation with a single device (no parallelism)."""
    mock_network.servers = [mock_network.servers[0]]  # Only one device

    result = simulate_alpa(
        model=mock_model,
        net=mock_network,
        seq_len=128,
        gen_tokens=10,
        bw_threshold_mbps=10.0,
    )

    assert result["policy"] == "alpa"
    assert len(result["clusters"]) == 1
    assert result["selected_plan"]["sharding"] == "none"
    assert result["selected_plan"]["per_layer_comm_s"] == 0.0
    assert "total_latency_s" in result


def test_simulate_alpa_client_server_model(
    mock_model, mock_network
):  # pylint: disable=redefined-outer-name
    """Test Alpa simulation with client-server communication model."""
    mock_network.communication_model = CommunicationModel.CLIENT_SERVER
    result = simulate_alpa(
        model=mock_model,
        net=mock_network,
        seq_len=128,
        gen_tokens=10,
        bw_threshold_mbps=10.0,
    )

    assert result["policy"] == "alpa"
    assert result["topology"] == "client_server"
    assert len(result["clusters"]) == 1
    assert len(result["clusters"][0]) == 2  # All devices in one cluster
    assert result["selected_plan"]["sharding"] in ["head", "col", "row"]


def test_alpa_form_clusters_high_bandwidth(
    network_multiple_devices: Network,
):  # pylint: disable=redefined-outer-name
    """Test Alpa cluster formation with high bandwidth threshold."""
    clusters = _alpa_form_clusters(network_multiple_devices, bw_threshold_mbps=60.0)

    assert len(clusters) == len(
        network_multiple_devices.servers
    ), "High bandwidth threshold should create as many clusters as devices"


def test_alpa_form_clusters_medium_bandwidth(
    network_multiple_devices: Network,
):  # pylint: disable=redefined-outer-name
    """Test Alpa cluster formation with medium bandwidth threshold."""
    clusters = _alpa_form_clusters(network_multiple_devices, bw_threshold_mbps=50.0)

    assert (
        len(clusters) > 1
    ), "Medium bandwidth threshold should create multiple clusters"


def test_alpa_form_clusters_low_bandwidth(
    network_multiple_devices: Network,
):  # pylint: disable=redefined-outer-name
    """Test Alpa cluster formation with low bandwidth threshold."""
    clusters = _alpa_form_clusters(network_multiple_devices, bw_threshold_mbps=5.0)

    assert len(clusters) == 1, "Low bandwidth threshold should create one cluster"


def test_simulate_alpa_peer_to_peer_high_bandwidth(
    mock_model, mock_network
):  # pylint: disable=redefined-outer-name
    """Test Alpa simulation with peer-to-peer communication model and high bandwidth."""
    mock_network.communication_model = CommunicationModel.PEER_TO_PEER
    mock_network.link.return_value = (50.0, 0.01)  # High bandwidth

    result = simulate_alpa(
        model=mock_model,
        net=mock_network,
        seq_len=128,
        gen_tokens=10,
        bw_threshold_mbps=50.0,
    )

    assert result["policy"] == "alpa"
    assert result["topology"] == "peer_to_peer"
    assert len(result["clusters"]) == 1  # High bandwidth forms one cluster
    assert len(result["clusters"][0]) == 2


def test_simulate_alpa_peer_to_peer_low_bandwidth(
    mock_model, mock_network
):  # pylint: disable=redefined-outer-name
    """Test Alpa simulation with peer-to-peer communication model and low bandwidth."""
    mock_network.communication_model = CommunicationModel.PEER_TO_PEER
    mock_network.link.return_value = (5.0, 0.01)  # Low bandwidth

    result = simulate_alpa(
        model=mock_model,
        net=mock_network,
        seq_len=128,
        gen_tokens=10,
        bw_threshold_mbps=10.0,
    )

    assert result["policy"] == "alpa"
    assert result["topology"] == "peer_to_peer"
    assert len(result["clusters"]) == 2  # Low bandwidth creates separate clusters


def test_simulate_alpa_head_parallel_assignment(
    mock_model, mock_network
):  # pylint: disable=redefined-outer-name
    """Test Alpa simulation selecting head-parallel sharding and correct head assignment."""
    mock_network.communication_model = CommunicationModel.CLIENT_SERVER
    mock_model.num_heads = 32
    mock_model.num_kv_heads = 8

    result = simulate_alpa(
        model=mock_model,
        net=mock_network,
        seq_len=128,
        gen_tokens=10,
    )

    selected_plan = result["selected_plan"]
    if selected_plan["sharding"] == "head":
        assert "q_heads" in selected_plan
        assert "kv_heads" in selected_plan
        assert sum(selected_plan["q_heads"]) == 32
        assert sum(selected_plan["kv_heads"]) == 8


def test_simulate_alpa_empty_network():  # pylint: disable=redefined-outer-name
    """Test Alpa simulation with an empty network (no devices)."""
    mock_model_empty = Mock(spec=ModelSpec)
    mock_network_empty = Mock(spec=Network)
    mock_network_empty.servers = []

    with pytest.raises(AssertionError, match="No devices available"):
        simulate_alpa(
            model=mock_model_empty, net=mock_network_empty, seq_len=128, gen_tokens=10
        )


def test_simulate_alpa_multiple_devices_different_performance():  # pylint: disable=redefined-outer-name
    """Test Alpa simulation with multiple devices of varying performance."""
    model = Mock(spec=ModelSpec)
    model.num_heads = 16
    model.num_kv_heads = 4
    model.num_layers = 12
    model.attn_allreduce_bytes = 512
    model.ffn_allreduce_bytes = 1024
    model.activation_bytes = 2048
    model.layer_compute_flops.return_value = 500000
    model.layer_compute_cost_per_device.return_value = [0.2, 0.1, 0.15]

    net = Mock(spec=Network)

    # Three devices with different performance
    device1 = Mock()
    device1.name = "gpu0"
    device1.flops_per_sec.return_value = 5000000

    device2 = Mock()
    device2.name = "gpu1"
    device2.flops_per_sec.return_value = 10000000

    device3 = Mock()
    device3.name = "gpu2"
    device3.flops_per_sec.return_value = 7500000

    net.servers = [device1, device2, device3]
    net.communication_model = CommunicationModel.PEER_TO_PEER
    net.link.return_value = (150.0, 0.005)
    net.ring_allreduce_time_with_pairwise_links.return_value = 0.0015
    net.allreduce_communication_cost.return_value = 0.0025

    result = simulate_alpa(
        model=model,
        net=net,
        seq_len=256,
        gen_tokens=5,
        bw_threshold_mbps=50.0,
    )

    assert result["policy"] == "alpa"
    assert len(result["selected_plan"]["devices"]) == 3
    assert result["selected_plan"]["sharding"] in ["head", "col", "row"]
    assert result["total_latency_s"] > 0


@pytest.mark.parametrize("sharding_type", ["head", "col", "row"])
def test_simulate_alpa_sharding_types(
    mock_model, mock_network, sharding_type
):  # pylint: disable=redefined-outer-name
    """Test Alpa simulation selecting different sharding types by mocking compute times."""
    mock_network.communication_model = CommunicationModel.CLIENT_SERVER
    # Mock the compute time functions to force a specific sharding choice
    with patch(
        "graph_partitioning.policies.alpa._layer_compute_time_head_parallel"
    ) as mock_hp, patch(
        "graph_partitioning.policies.alpa._layer_compute_time_column_parallel"
    ) as mock_cp, patch(
        "graph_partitioning.policies.alpa._layer_compute_time_row_parallel"
    ) as mock_rp:

        if sharding_type == "head":
            mock_hp.return_value = 0.05
            mock_cp.return_value = 0.1
            mock_rp.return_value = 0.1
        elif sharding_type == "col":
            mock_hp.return_value = 0.1
            mock_cp.return_value = 0.05
            mock_rp.return_value = 0.1
        else:  # row
            mock_hp.return_value = 0.1
            mock_cp.return_value = 0.1
            mock_rp.return_value = 0.05

        result = simulate_alpa(
            model=mock_model,
            net=mock_network,
            seq_len=128,
            gen_tokens=10,
        )

        assert result["selected_plan"]["sharding"] == sharding_type
