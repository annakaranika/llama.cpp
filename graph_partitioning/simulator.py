"""Simulator for distributed LLM inference across heterogeneous devices.

Supports multiple partitioning policies and strategies and communication models.
Exposes a simple interface for running multiple experiments and logging/visualizing results.
"""

from typing import Callable, Dict, Optional, Union

import json
import logging
import os


from .common import (
    CommunicationModel,
    PeerToPeerPolicy,
    AttentionSplitStrategy,
    FFNSplitStrategy,
    PartitioningPolicy,
    MetisDevConsideration,
    DevNetAggregationStrategy,
)
from .compare_policies import plot_all_policies
from .graph_plots import set_output_dir
from .logging_utils import make_json_safe, setup_logger
from .model import ModelSpec
from .network import Network
from .policies.single_node import simulate_single_node
from .policies.pipeline import simulate_pipeline
from .policies.tensor import simulate_tensor_parallel
from .policies.metis import simulate_metis
from .policies.alpa import simulate_alpa
from .policies.hybrid_pp_tp import simulate_hybrid_pp_tp


class Simulator:
    """Simulator for distributed LLM inference across heterogeneous devices."""

    PARTITIONING_POLICY_MAP: Dict[Union[str, PartitioningPolicy], Callable] = {
        PartitioningPolicy.SINGLE_NODE: simulate_single_node,
        PartitioningPolicy.PIPELINE: simulate_pipeline,
        PartitioningPolicy.TENSOR: simulate_tensor_parallel,
        PartitioningPolicy.METIS: simulate_metis,
        PartitioningPolicy.ALPA: simulate_alpa,
        PartitioningPolicy.HYBRID_PP_TP: simulate_hybrid_pp_tp,
    }

    # Automatically add string value keys pointing to the same callables
    PARTITIONING_POLICY_MAP.update(
        {
            policy.value: func
            for policy, func in PARTITIONING_POLICY_MAP.items()
            if isinstance(policy, PartitioningPolicy)
        }
    )

    model: ModelSpec
    net: Network
    output_dir: Optional[str] = None
    exp_logger: logging.Logger

    def __init__(
        self,
        net: Optional[Network] = None,
        model: Optional[ModelSpec] = None,
        output_dir: Optional[str] = None,
    ):
        self.net = net if net is not None else Network()
        self.model = model if model is not None else ModelSpec()
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            # Use a per-instance logger name so re-running this cell with a
            # fresh output_dir doesn't get short-circuited by setup_logger's
            # name-based cache (which would otherwise return the previous
            # logger and skip attaching the new FileHandler).
            self.exp_logger = setup_logger(
                name=f"simulator.{os.path.basename(self.output_dir.rstrip('/'))}",
                log_file=os.path.join(self.output_dir, "simulator.log"),
            )
            set_output_dir(self.output_dir)
        else:
            self.exp_logger = setup_logger(name=__name__)

    def compare_tensor_parallel_strategies(
        self,
        seq_len: int,
        gen_tokens: int,
    ) -> Dict[str, float]:
        """Compare the three tensor parallel splitting strategies."""

        self.exp_logger.info("=" * 60)
        self.exp_logger.info("Comparing tensor parallel strategies")
        self.exp_logger.info("=" * 60)

        self.exp_logger.info(
            "Model: %s (%s layers)", self.model.name, self.model.num_layers
        )
        self.exp_logger.info(
            "Devices: %s", [d.name + f"({d.gflops}GF)" for d in self.net.servers]
        )
        self.exp_logger.info(
            "Workload: %s context + %s generation tokens", seq_len, gen_tokens
        )

        strategies = [
            (
                AttentionSplitStrategy.KV_HEADS,
                FFNSplitStrategy.KV_HEADS,
            ),  # Both split by KV head groups - most aligned
            (
                AttentionSplitStrategy.KV_HEADS,
                FFNSplitStrategy.HIDDEN_DIM,
            ),  # Attention by KV heads, FFN traditional split
            (
                AttentionSplitStrategy.KV_HEADS,
                FFNSplitStrategy.Q_HEADS,
            ),  # Attention by KV heads, FFN by Q head groups
        ]

        latencies = {}
        for attn_strategy, ffn_strategy in strategies:
            strategy_name = f"attn_{attn_strategy.name}_ffn_{ffn_strategy.name}"
            result = simulate_metis(
                self.model,
                self.net,
                seq_len,
                gen_tokens,
                attn_split_strategy=attn_strategy,
                ffn_split_strategy=ffn_strategy,
            )
            latencies[strategy_name] = result["total_latency_s"]
            self.exp_logger.info(
                "Experiment result:\n%s",
                # pformat(results, width=100, depth=3),
                json.dumps(make_json_safe(result), indent=2, default=str),
            )

        return latencies

    def compare_comm_configs_for_strategy(
        self,
        policy_name: Union[str, PartitioningPolicy],
        seq_len: int,
        gen_tokens: int,
    ) -> Dict[str, float]:
        """Compare the three communication models."""

        self.exp_logger.info("=" * 60)
        self.exp_logger.info(
            "Comparing communication configurations for strategy: %s", policy_name
        )
        self.exp_logger.info("=" * 60)

        self.exp_logger.info(
            "Model: %s (%s layers, %.2f GFLOPS, %.2f GB memory)",
            self.model.name,
            self.model.num_layers,
            self.model.total_compute_flops() / 1e9,
            self.model.total_memory_usage(),
        )
        self.exp_logger.info(
            "Devices: %s",
            [d.name + f"({d.gflops}GF+{d.memory_gb}GB)" for d in self.net.servers],
        )
        self.exp_logger.info(
            "Workload: %s context + %s generation tokens", seq_len, gen_tokens
        )

        combinations = [(CommunicationModel.CLIENT_SERVER, None)] + [
            (CommunicationModel.PEER_TO_PEER, p2p_policy)
            for p2p_policy in PeerToPeerPolicy
            if p2p_policy is not PeerToPeerPolicy.HIERARCHICAL
        ]

        latencies = {}
        for comm_model, p2p_policy in combinations:
            model_name = comm_model.name.lower() + (
                f"_{p2p_policy.name.lower()}" if p2p_policy else ""
            )
            self.net.set_comm_config(comm_model, p2p_policy)
            result = self.PARTITIONING_POLICY_MAP[policy_name](
                self.model,
                self.net,
                seq_len,
                gen_tokens,
            )
            latencies[model_name] = result["total_latency_s"]

            self.exp_logger.info(
                "Experiment result:\n%s",
                # pformat(results, width=100, depth=None),
                json.dumps(make_json_safe(result), indent=2, default=str),
            )

        return latencies

    def compare_sched_policies(
        self,
        seq_len: int = 512,
        gen_tokens: int = 64,
    ) -> Dict[str, float]:
        """
        Run all policies on the given device set.
        Returns a dict of results keyed by policy.
        """
        # Setup logger for this experiment run
        self.exp_logger.info("=" * 60)
        self.exp_logger.info("Starting LLM IoT simulation experiments")
        self.exp_logger.info("=" * 60)

        self.exp_logger.info(
            "Model: %s (%s layers, %.2f GFLOPS, %.2f GB memory)",
            self.model.name,
            self.model.num_layers,
            self.model.total_compute_flops() / 1e9,
            self.model.total_memory_usage(),
        )
        self.exp_logger.info(
            "Devices: %s",
            [
                d.repr(self.net.aggregate_bandwidth_capacity(d))
                for d in self.net.servers
            ],
        )
        self.exp_logger.info(
            "Workload: %s context + %s generation tokens", seq_len, gen_tokens
        )

        latencies = {}
        for policy in PartitioningPolicy:
            if policy == PartitioningPolicy.ALPA:
                continue  # ALPA is handled separately

            self.exp_logger.info("=" * 60)
            self.exp_logger.info("Running experiment with policy: %s", policy)
            self.exp_logger.info("=" * 60)

            result = self.PARTITIONING_POLICY_MAP[policy](
                self.model, self.net, seq_len, gen_tokens
            )

            latencies[policy.value] = result["total_latency_s"]

            self.exp_logger.info(
                "Experiment result:\n%s",
                json.dumps(make_json_safe(result), indent=2, default=str),
            )

        return latencies

    def compare_dev_policies(
        self,
        seq_len: int,
        gen_tokens: int,
    ) -> Dict:
        """Run a single experiment with the given policy."""

        self.exp_logger.info("=" * 60)
        self.exp_logger.info("Comparing device configurations with METIS")
        self.exp_logger.info("=" * 60)

        latencies = {}

        for dev_net_aggr in DevNetAggregationStrategy:
            for dev_cons in MetisDevConsideration:
                if (
                    dev_cons
                    in [
                        MetisDevConsideration.COMPUTE_ONLY,
                        MetisDevConsideration.MEMORY_ONLY,
                    ]
                    and dev_net_aggr != DevNetAggregationStrategy.SUM
                ):
                    # One aggregation is enough for non-network-based considerations (no effect)
                    continue
                self.exp_logger.info(
                    "Device consideration: %s, Network aggregation: %s",
                    dev_cons,
                    dev_net_aggr,
                )

                self.net.set_dev_net_aggregation_strategy(dev_net_aggr)

                result = simulate_metis(
                    self.model,
                    self.net,
                    seq_len,
                    gen_tokens,
                    dev_consideration=dev_cons,
                )
                latencies[dev_cons.value + "_" + dev_net_aggr.value] = result[
                    "total_latency_s"
                ]

                self.exp_logger.info(
                    "Experiment result:\n%s",
                    json.dumps(make_json_safe(result), indent=2, default=str),
                )

        return latencies

    def plot_all_partitions(
        self,
        seq_len: int = 512,
        gen_tokens: int = 64,
        attn_split: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
        ffn_split: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
    ) -> Dict[str, Dict]:
        """Run every policy and emit a unified set of comparison plots.

        Produces, under ``<self.output_dir>/compare`` (or a fresh timestamped
        dir if no output_dir is set):
          - ``policy_partition_heatmaps.pdf``
          - ``policy_latency_comparison.pdf``
          - ``policy_device_workload_share.pdf``
          - ``policy_results.json``
        """
        self.exp_logger.info("=" * 60)
        self.exp_logger.info("Plotting partitions for all policies")
        self.exp_logger.info("=" * 60)

        compare_dir = (
            os.path.join(self.output_dir, "compare") if self.output_dir else None
        )
        if compare_dir:
            os.makedirs(compare_dir, exist_ok=True)

        results, used_dir = plot_all_policies(
            self.model,
            self.net,
            seq_len,
            gen_tokens,
            output_dir=compare_dir,
            attn_split=attn_split,
            ffn_split=ffn_split,
        )
        self.exp_logger.info("Comparison plots written to %s", used_dir)
        return results

    def evaluate_ttft_tpt(
        self,
        prompt_lengths: Optional[list[int]] = None,
    ) -> dict:
        """Evaluate TTFT (Time to First Token) and TPT (Time Per Token) across policies.

        Saves output plots to the simulator's output directory.
        """
        self.exp_logger.info("=" * 60)
        self.exp_logger.info("Evaluating TTFT and TPT for all policies")
        self.exp_logger.info("=" * 60)

        output_dir = (
            os.path.join(self.output_dir, "latency") if self.output_dir else "sim_output/latency"
        )
        os.makedirs(output_dir, exist_ok=True)

        from .evaluate import run_latency_evaluation
        results = run_latency_evaluation(
            self.net,
            self.model,
            prompt_lengths=prompt_lengths,
            output_dir=output_dir,
        )
        self.exp_logger.info("Latency evaluation complete. Outputs in %s", output_dir)
        return results
