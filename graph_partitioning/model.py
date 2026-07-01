"""Model specification and cost modeling for LLM inference simulation."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


import os

from .common import AttentionSplitStrategy, Device, FFNSplitStrategy, Quantization
from .logging_utils import setup_logger
from .network import Network

os.makedirs("sim_output", exist_ok=True)
LOGGER = setup_logger(name=__name__, log_file="sim_output/model.log")


@dataclass
class ModelSpec:
    """
    Model specification class for LLM inference simulation.

    This class defines the architectural parameters and cost modeling coefficients
    for a language model, specifically configured for TinyLlama-1.1B-Chat-v1.0.
    It includes parameters for transformer architecture, communication patterns,
    and performance estimation coefficients.

    Attributes:
        name (str): Model identifier, defaults to "TinyLlama-1.1B-Chat-v1.0"
        num_layers (int): Total number of transformer layers including output layer
        num_heads (int): Number of attention heads in multi-head attention
        num_kv_heads (int): Number of key-value heads for grouped query attention
        d_model (int): Model dimensionality/hidden size
        d_k (int): Dimension per attention head
        d_ff (int): Feed-forward network hidden dimension
        vocab_size (int): Size of the model's vocabulary
        activation_bytes (int): Activation transfer size in bytes per token at layer boundaries
        attn_allreduce_bytes (int): Collective communication payload per token per layer for attention  # pylint: disable=line-too-long
        ffn_allreduce_bytes (int): Collective communication payload per token per layer for FFN
        output_allreduce_bytes (int): Collective communication payload per token per layer for output logits
        kv_bytes_per_head (int): KV cache memory usage per token per layer per head

    Methods:
        layer_compute_flops_per_token(seq_len): Estimates FLOPs per token per layer
            given sequence length, accounting for both linear and quadratic attention costs.

    Note:
        Coefficients are rough estimates and should be calibrated against actual measurements
        for accurate performance modeling.
    """

    name: str = "TinyLlama-1.1B-Chat-v1.0"
    num_layers: int = 22  # transformer blocks only (TinyLlama-1.1B has 22); the
    # output head (-ngl's +1) is a single projection counted via embedding_weights_gb()
    num_heads: int = 32
    num_kv_heads: int = 4
    d_model: int = 2048  # vector size
    d_k: int = 64  # per-head
    d_ff: int = 5632  # hidden size
    vocab_size: int = 32000

    # Quantization of stored weights and of the KV cache. These govern the
    # memory *footprint* (and hence the swap-penalty model) only — compute is
    # modeled separately via effective device GFLOPS. TinyLlama as actually
    # run here is Q5_K_M weights with an FP16 KV cache; the FP32 defaults
    # elsewhere in this file (activation/all-reduce payloads) are runtime
    # tensors on the wire, a separate concern from stored-weight size.
    weight_quant: Quantization = Quantization.Q5_K_M
    kv_quant: Quantization = Quantization.FP16

    # Derived byte sizes. Declared as Optional[int] = None so the values
    # are recomputed in __post_init__ from the instance's d_model / d_k /
    # vocab_size; using the literal expression here would freeze them to
    # the class-level defaults (e.g. d_model=2048) and silently break any
    # ModelSpec built with custom dimensions.
    #
    # NOTE: activation_bytes / *_allreduce_bytes are network payloads of
    # *runtime activations* (FP32 on CPU llama.cpp), independent of how the
    # weights are quantized — so they stay at 4 bytes/element.
    activation_bytes: Optional[int] = None
    attn_allreduce_bytes: Optional[int] = None
    ffn_allreduce_bytes: Optional[int] = None
    output_allreduce_bytes: Optional[int] = None
    kv_bytes_per_head: Optional[int] = None

    def __post_init__(self):
        if self.activation_bytes is None:
            self.activation_bytes = 4 * self.d_model
        if self.attn_allreduce_bytes is None:
            self.attn_allreduce_bytes = 4 * self.d_model
        if self.ffn_allreduce_bytes is None:
            self.ffn_allreduce_bytes = 4 * self.d_model
        if self.output_allreduce_bytes is None:
            self.output_allreduce_bytes = 4 * self.vocab_size
        if self.kv_bytes_per_head is None:
            # K and V, each kv_quant bytes per element, d_k elements per head.
            self.kv_bytes_per_head = 2 * self.kv_quant.bytes_per_param * self.d_k

    def attn_compute_flops(
        self,
        seq_len: int,
        dev_q_heads: Optional[int] = None,
        dev_kv_heads: Optional[int] = None,
    ) -> float:
        """
        Compute attention compute flops given sequence length and heads.

        Attention Architecture Flow:
        Input:  [batch, seq_len, d_model]                       # 2048 dim
        ↓
        Q projection:  d_model → num_heads    * d_k             # 2048 → 32 * 64 = 2048
        K projection:  d_model → num_kv_heads * d_k             # 2048 →  4 * 64 = 256
        V projection:  d_model → num_kv_heads * d_k             # 2048 →  4 * 64 = 256
        ↓
        Reshape: Q=[batch, seq_len, num_heads, d_k]             # [B, 512, 32, 64]
                 K=[batch, seq_len, num_kv_heads, d_k]          # [B, 512,  4, 64]
                 V=[batch, seq_len, num_kv_heads, d_k]          # [B, 512,  4, 64]
        ↓
        Expand K,V: K=[batch, seq_len, num_heads, d_k]          # [B, 512, 32, 64] (GQA: repeat each KV head 8x)  # pylint: disable=line-too-long
                    V=[batch, seq_len, num_heads, d_k]          # [B, 512, 32, 64]
        ↓
        Attention scores: Q @ K^T                               # [32, 512, 64] @ [32, 64, 512] = [32, 512, 512]  # pylint: disable=line-too-long
        ↓                                                       # seq_len² scaling! Memory + compute intensive  # pylint: disable=line-too-long
        Scale: scores / √d_k                                    # [32, 512, 512] / 8.0
        ↓
        Softmax: softmax(scores)                                # [32, 512, 512] row-wise softmax
        ↓
        Weighted sum: softmax_scores @ V                        # [32, 512, 512] @ [32, 512, 64] = [32, 512, 64]  # pylint: disable=line-too-long
        ↓
        Concatenate heads: [batch, seq_len, num_heads * d_k]    # [B, 512, 32*64] = [B, 512, 2048]
        ↓
        Output projection: (num_heads * d_k) → d_model          # 2048 → 2048
        ↓
        Output: [batch, seq_len, d_model]                       # 2048 dim (same as input)

        Key computational costs:
        - Linear projections: O(seq_len * d_model²)             # Q,K,V,O projections
        - Attention matrix: O(seq_len² * d_model)               # QK^T + softmax + attn@V
        - Memory: O(seq_len² * num_heads) for scores            # Attention matrix storage

        GQA (Grouped Query Attention) optimization:
        - num_kv_heads < num_heads (4 vs 32 in TinyLlama)
        - Each KV head serves multiple Q heads (8 Q heads per KV head)
        - Reduces KV cache memory: 4x instead of 32x storage
        - Same attention quality with less memory bandwidth
        """
        device_q_heads = dev_q_heads if dev_q_heads is not None else self.num_heads
        device_kv_heads = (
            dev_kv_heads if dev_kv_heads is not None else self.num_kv_heads
        )

        # Attention FLOPs based on head allocation

        # All matmuls below count 2 FLOPs per multiply-accumulate (MAC).
        # Q projection: [seq_len, d_model] @ [d_model, device_q_heads * d_k]
        q_proj_flops = 2 * seq_len * self.d_model * (device_q_heads * self.d_k)

        # K AND V projections (two matrices): [seq_len, d_model] @ [d_model, device_kv_heads * d_k]
        kv_proj_flops = 2 * 2 * seq_len * self.d_model * (device_kv_heads * self.d_k)

        # Attention computation: QK^T and attn@V, each 2 * seq_len^2 * (q_heads * d_k)
        # (GQA: K/V are broadcast to the Q heads on this device). Softmax is negligible.
        attn_compute_flops = 2 * 2 * (seq_len**2) * (device_q_heads * self.d_k)

        # Output projection: [seq_len, device_q_heads * d_k] @ [device_q_heads * d_k, d_model]
        out_proj_flops = 2 * seq_len * self.d_model * (device_q_heads * self.d_k)

        total_attn_flops = (
            q_proj_flops + kv_proj_flops + attn_compute_flops + out_proj_flops
        )
        return total_attn_flops

    def ffn_compute_flops(self, seq_len: int, ffn_fraction: float = 1.0) -> float:
        """
        Compute FFN compute flops given FFN fraction.
        FFN has two linear layers: d_model → d_ff → d_model (up + down proj)

        Input:  [batch, seq_len, d_model]     # 2048 dim
        ↓
        Up projection:   d_model → d_ff       # 2048 → 5632
        ↓
        Activation (GeLU, SwiGLU, etc.)       # elementwise, usually negligible
        ↓
        Down projection: d_ff → d_model       # 5632 → 2048
        ↓
        Output: [batch, seq_len, d_model]     # 2048 dim (same as input)
        """
        # SwiGLU FFN has THREE matrices: gate and up (both d_model -> d_ff) and
        # down (d_ff -> d_model). 2 FLOPs per MAC. gate and up are the same shape.
        gate_up_flops = 2 * 2 * seq_len * self.d_model * self.d_ff * ffn_fraction

        # Down projection: [seq_len, d_ff] @ [d_ff, d_model]
        down_proj_flops = 2 * seq_len * self.d_ff * self.d_model * ffn_fraction

        return gate_up_flops + down_proj_flops

    def layer_compute_flops(
        self,
        seq_len: int = 1,
        dev_q_heads: Optional[int] = None,
        dev_kv_heads: Optional[int] = None,
        ffn_fraction: float = 1.0,
    ) -> float:
        """
        Very rough per-layer compute FLOPs for generating one token given context length seq_len.
        """
        attn = self.attn_compute_flops(
            seq_len,
            dev_q_heads=dev_q_heads,
            dev_kv_heads=dev_kv_heads,
        )
        ffn = self.ffn_compute_flops(seq_len, ffn_fraction=ffn_fraction)
        return attn + ffn

    def total_compute_flops(self) -> float:
        """
        Total compute FLOPs for the entire model (all layers) for generating one token.
        Assumes full model is used without any splits.
        """
        per_layer_flops = self.layer_compute_flops()
        total_flops = per_layer_flops * self.num_layers
        return total_flops

    def layer_compute_cost_per_device(
        self,
        seq_len: int,
        devices: List[Device],
        device_q_counts: Optional[List[int]] = None,
        device_kv_counts: Optional[List[int]] = None,
        device_ffn_fractions: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Compute per-layer compute times across a list of devices
        given their attention and FFN allocations.
        Returns the total compute time (s) for the layer across all devices,
        as well as compute time standard deviation and load balance efficiency.
        """

        per_layer_compute_times = []
        memory_usages = self.memory_usage_per_device(
            seq_len,
            devices,
            num_layers=[1] * len(devices),
            device_kv_counts=device_kv_counts,
            device_ffn_fractions=device_ffn_fractions,
        )

        for i, device in enumerate(devices):
            device_q_heads = device_q_counts[i] if device_q_counts else self.num_heads
            device_kv_heads = (
                device_kv_counts[i] if device_kv_counts else self.num_kv_heads
            )
            device_ffn_fraction = (
                device_ffn_fractions[i] if device_ffn_fractions else 1.0
            )

            attn_flops = self.attn_compute_flops(
                seq_len,
                dev_q_heads=device_q_heads,
                dev_kv_heads=device_kv_heads,
            )

            ffn_flops = self.ffn_compute_flops(
                seq_len, ffn_fraction=device_ffn_fraction
            )

            total_flops = attn_flops + ffn_flops
            compute_time = device.compute_time_s(total_flops, memory_usages[i])
            per_layer_compute_times.append(compute_time)

        return per_layer_compute_times

    def attn_weights_gb_per_kv_head(self) -> float:
        """Weights-only attention memory (GB) per KV head, independent of seq_len.

        Counts all four projections of a GQA attention block, amortised per KV
        head: Q and the output projection O both span q_heads_per_kv heads,
        while K and V are a single head each.
        """
        q_heads_per_kv = self.num_heads // self.num_kv_heads
        wb = self.weight_quant.bytes_per_param
        attn_weights = (
            q_heads_per_kv * self.d_k * self.d_model * wb  # Q proj
            + self.d_k * self.d_model * 2 * wb  # K, V projections
            + q_heads_per_kv * self.d_k * self.d_model * wb  # O (output) proj
        )
        return attn_weights / 1e9  # decimal GB (matches GB/s bandwidth units)

    def kv_cache_gb_per_kv_head(self, seq_len: int) -> float:
        """KV cache memory (GB) per KV head at the given seq_len."""
        return (self.kv_bytes_per_head * seq_len) / 1e9

    def attn_memory_gb_per_kv_head(self, seq_len: int = 512) -> float:
        """Attention working set (weights + KV cache at ``seq_len``) per KV head."""
        return self.attn_weights_gb_per_kv_head() + self.kv_cache_gb_per_kv_head(
            seq_len
        )

    def ffn_memory_gb_per_fraction(self, ffn_fraction: float) -> float:
        """FFN weights memory (GB) for a given FFN split fraction, no seq_len.

        TinyLlama/Llama use a SwiGLU FFN with THREE matrices — gate, up
        (both d_model x d_ff) and down (d_ff x d_model) — hence the factor 3.
        """
        wb = self.weight_quant.bytes_per_param
        ffn_weights = ffn_fraction * self.d_model * self.d_ff * 3 * wb
        return ffn_weights / 1e9

    def kv_cache_gb(
        self,
        seq_len: int,
        device_kv_counts: Optional[List[int]] = None,
        num_layers: Optional[List[int]] = None,
    ) -> List[float]:
        """KV cache memory (GB) per device for the given context length.

        Decoupled from ``memory_usage_per_device`` so callers (and the
        adaptive scheduler) can reason explicitly about cache growth vs.
        static weight footprint.
        """
        n = len(device_kv_counts) if device_kv_counts else 1
        kv_counts = device_kv_counts if device_kv_counts else [self.num_kv_heads] * n
        layers = num_layers if num_layers else [self.num_layers] * n
        per_head = self.kv_cache_gb_per_kv_head(seq_len)
        return [per_head * kv_counts[i] * layers[i] for i in range(n)]

    def memory_usage_per_device(
        self,
        seq_len: int = 512,
        devices: Optional[List[Device]] = None,
        num_layers: Optional[List[int]] = None,
        device_kv_counts: Optional[List[int]] = None,
        device_ffn_fractions: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Memory footprint (GB) per device = weights + KV cache at ``seq_len``.

        ``seq_len`` here is the working KV-cache length — for prefill it's the
        prompt length, for decode it's prompt + tokens decoded so far.
        ``num_layers`` is the per-device layer count (defaults to the full
        model on every device, i.e. no pipeline split).
        """
        if devices is None:
            devices = [Device(name="default", gflops=0.0, memory_gb=0.0)]
        memory_usages = []
        for i in range(len(devices)):
            kv_heads = device_kv_counts[i] if device_kv_counts else self.num_kv_heads
            attn_memory_gb = self.attn_memory_gb_per_kv_head(seq_len) * kv_heads
            ffn_fraction = device_ffn_fractions[i] if device_ffn_fractions else 1.0
            ffn_weights = self.ffn_memory_gb_per_fraction(ffn_fraction)
            dev_num_layers = num_layers[i] if num_layers else self.num_layers
            total_memory_gb = (attn_memory_gb + ffn_weights) * dev_num_layers
            memory_usages.append(total_memory_gb)
        return memory_usages

    def embedding_weights_gb(self) -> float:
        """Token-embedding / output-projection weights (GB).

        A single vocab x d_model matrix; TinyLlama ties the input embedding and
        the output (lm_head), so it is counted once. Lives outside the
        per-layer transformer stack — in a pipeline split it sits on the
        first/last stage rather than being shared across all of them.
        """
        wb = self.weight_quant.bytes_per_param
        return (self.vocab_size * self.d_model * wb) / 1e9

    def total_memory_usage(self, seq_len: int = 512) -> float:
        """Total model memory (GB) at the given KV-cache length, no splits.

        Includes the full transformer stack plus the embedding/output matrix,
        so it reflects the real on-device footprint (e.g. ~0.73 GB for
        TinyLlama-1.1B Q5_K_M) that drives the swap-penalty model.
        """
        total_memory = self.memory_usage_per_device(seq_len=seq_len)[0]
        return total_memory + self.embedding_weights_gb()

    def get_max_attn_splits(self, strategy: AttentionSplitStrategy) -> int:
        """Get maximum number of attention splits for given strategy."""
        return self.num_kv_heads if strategy == AttentionSplitStrategy.KV_HEADS else 1

    def get_max_ffn_splits(self, strategy: FFNSplitStrategy, num_devices: int) -> int:
        """Get maximum number of FFN splits for given strategy."""
        return {
            FFNSplitStrategy.KV_HEADS: self.num_kv_heads,
            FFNSplitStrategy.Q_HEADS: self.num_heads,
            FFNSplitStrategy.HIDDEN_DIM: num_devices,  # Can split arbitrarily across hidden dim
        }[strategy]

    def build_computation_graph(
        self,
        net: Network,
        seq_len: int,
        attn_split_strategy: AttentionSplitStrategy = AttentionSplitStrategy.KV_HEADS,
        ffn_split_strategy: FFNSplitStrategy = FFNSplitStrategy.KV_HEADS,
        device_compute_callback: Optional[
            Callable[[int, int, int, int], Tuple[float, float]]
        ] = None,
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
        """
        Build a computation graph with separate attention and FFN blocks per layer.

        Topology per layer (star representation of the tensor-parallel all-reduce):
        - ``max_attn_splits`` attention split nodes, each connected to a single
          virtual ``attn_AR`` node with weight = full attention all-reduce cost
        - ``max_ffn_splits`` FFN split nodes, each connected to a single virtual
          ``ffn_AR`` node with weight = full FFN all-reduce cost
        - One edge ``attn_AR -> ffn_AR`` (intra-layer activation transfer)
        - One edge ``ffn_AR[L] -> attn_AR[L+1]`` (inter-layer activation transfer)

        The virtual all-reduce nodes have negligible compute weight so METIS
        does not try to balance compute by placing them; their role is to
        carry the full all-reduce cost on any cut that separates a split from
        its peers. Every edge is emitted symmetrically on both endpoints,
        which is required for METIS's undirected-graph interpretation.

        Args:
            net: Network topology
            seq_len: Sequence length
            attn_split_strategy: How to split attention computation
            ffn_split_strategy: How to split FFN computation
            device_compute_callback: Optional callback to compute device-specific costs
                Format: (layer, block_type, split_id, max_splits) -> (attn_flops, ffn_flops)
                If None, uses default equal split computation

        Returns:
            - xadj: Adjacency list starts
            - adjncy: Adjacent nodes
            - eweights: Edge weights
            - node_types: 0=attn split, 1=ffn split, 2=attn_AR virtual, 3=ffn_AR virtual
            - node_splits: Which split each node represents (-1 for virtual nodes)
            - node_weights: Node weights based on computation flops
        """

        LOGGER.info("Building computation graph for %s", self.name)
        LOGGER.info(
            "Strategies: attention=%s, ffn=%s",
            attn_split_strategy.name,
            ffn_split_strategy.name,
        )

        k = len(net.servers)
        n_layers = self.num_layers
        ref_gflops = (
            sum(s.gflops for s in net.servers) / len(net.servers)
            if net.servers
            else 1.0
        )

        max_attn_splits = self.get_max_attn_splits(attn_split_strategy)
        max_ffn_splits = self.get_max_ffn_splits(ffn_split_strategy, k)
        # Node layout per layer:
        #   [0 .. max_attn_splits)                   -> attn splits
        #   [max_attn_splits .. ATTN_VIRT_OFFSET)    -> ffn splits
        #   ATTN_VIRT_OFFSET                         -> attn all-reduce virtual node
        #   FFN_VIRT_OFFSET                          -> ffn all-reduce virtual node
        ATTN_VIRT_OFFSET = max_attn_splits + max_ffn_splits
        FFN_VIRT_OFFSET = ATTN_VIRT_OFFSET + 1
        nodes_per_layer = FFN_VIRT_OFFSET + 1
        total_nodes = n_layers * nodes_per_layer

        # Type sentinels
        TYPE_ATTN_SPLIT = 0
        TYPE_FFN_SPLIT = 1
        TYPE_ATTN_VIRT = 2
        TYPE_FFN_VIRT = 3

        def attn_split_node(layer: int, split_id: int) -> int:
            return layer * nodes_per_layer + split_id

        def ffn_split_node(layer: int, split_id: int) -> int:
            return layer * nodes_per_layer + max_attn_splits + split_id

        def attn_virt_node(layer: int) -> int:
            return layer * nodes_per_layer + ATTN_VIRT_OFFSET

        def ffn_virt_node(layer: int) -> int:
            return layer * nodes_per_layer + FFN_VIRT_OFFSET

        # Build node compute weights and type metadata.
        node_weights: List[int] = [0] * total_nodes
        node_types: List[int] = [0] * total_nodes
        split_ids: List[int] = [0] * total_nodes

        for layer in range(n_layers):
            # Attention splits
            for split_id in range(max_attn_splits):
                if device_compute_callback:
                    compute_flops, _ = device_compute_callback(
                        layer, TYPE_ATTN_SPLIT, split_id, max_attn_splits
                    )
                else:
                    dev_kv_heads = self.num_kv_heads // max_attn_splits
                    dev_q_heads = (
                        self.num_heads // self.num_kv_heads
                    ) * dev_kv_heads
                    compute_flops = self.attn_compute_flops(
                        seq_len,
                        dev_q_heads=dev_q_heads,
                        dev_kv_heads=dev_kv_heads,
                    )
                nid = attn_split_node(layer, split_id)
                node_weights[nid] = max(
                    1, int(compute_flops / (ref_gflops * 1000.0))
                )
                node_types[nid] = TYPE_ATTN_SPLIT
                split_ids[nid] = split_id

            # FFN splits
            for split_id in range(max_ffn_splits):
                if device_compute_callback:
                    _, compute_flops = device_compute_callback(
                        layer, TYPE_FFN_SPLIT, split_id, max_ffn_splits
                    )
                else:
                    ffn_fraction = 1.0 / max_ffn_splits
                    compute_flops = self.ffn_compute_flops(
                        seq_len, ffn_fraction=ffn_fraction
                    )
                nid = ffn_split_node(layer, split_id)
                node_weights[nid] = max(
                    1, int(compute_flops / (ref_gflops * 1000.0))
                )
                node_types[nid] = TYPE_FFN_SPLIT
                split_ids[nid] = split_id

            # Virtual all-reduce nodes: weight 1 (negligible) so METIS does
            # not over-balance compute on them, but non-zero so they always
            # participate in the partitioning.
            av = attn_virt_node(layer)
            fv = ffn_virt_node(layer)
            node_weights[av] = 1
            node_weights[fv] = 1
            node_types[av] = TYPE_ATTN_VIRT
            node_types[fv] = TYPE_FFN_VIRT
            split_ids[av] = -1
            split_ids[fv] = -1

        # Edge construction. Build an undirected adjacency dict and then
        # flatten to CSR so every edge is emitted on both endpoints (METIS
        # requires symmetric adjacency).
        adj: List[List[Tuple[int, int]]] = [[] for _ in range(total_nodes)]

        def add_edge(u: int, v: int, weight: int) -> None:
            adj[u].append((v, weight))
            adj[v].append((u, weight))

        attn_ar_cost_us = max(
            1,
            int(
                net.allreduce_communication_cost(self.attn_allreduce_bytes)
                * 1_000_000.0
            ),
        )
        ffn_ar_cost_us = max(
            1,
            int(
                net.allreduce_communication_cost(self.ffn_allreduce_bytes)
                * 1_000_000.0
            ),
        )
        intra_layer_xfer_us = max(
            1,
            int(
                net.avg_server_server_time(self.activation_bytes) * 1_000_000.0
            ),
        )
        inter_layer_xfer_us = intra_layer_xfer_us

        for layer in range(n_layers):
            av = attn_virt_node(layer)
            fv = ffn_virt_node(layer)

            # Fan-IN: each attention split sends its partial sum into attn_AR
            # for the all-reduce. Weight is the full all-reduce cost so any
            # cut that puts a split on a different device than attn_AR pays
            # the full collective.
            for split_id in range(max_attn_splits):
                add_edge(attn_split_node(layer, split_id), av, attn_ar_cost_us)

            # Fan-OUT: after the all-reduce, the consolidated attention output
            # must be broadcast to every FFN split in the same layer. If
            # an FFN split is on a different device than attn_AR, an
            # activation transfer is paid.
            for split_id in range(max_ffn_splits):
                add_edge(av, ffn_split_node(layer, split_id), intra_layer_xfer_us)

            # Fan-IN: FFN splits all-reduce into ffn_AR.
            for split_id in range(max_ffn_splits):
                add_edge(ffn_split_node(layer, split_id), fv, ffn_ar_cost_us)

            # Fan-OUT (inter-layer): the consolidated FFN output is consumed
            # by every attention split in the next layer.
            if layer < n_layers - 1:
                for split_id in range(max_attn_splits):
                    add_edge(
                        fv,
                        attn_split_node(layer + 1, split_id),
                        inter_layer_xfer_us,
                    )

        xadj: List[int] = [0]
        adjncy: List[int] = []
        eweights: List[int] = []
        for nid in range(total_nodes):
            for neighbor, weight in adj[nid]:
                adjncy.append(neighbor)
                eweights.append(weight)
            xadj.append(len(adjncy))

        LOGGER.info(
            "Graph construction complete: %d nodes "
            "(%d compute + %d virtual all-reduce), %d directed edge entries",
            total_nodes,
            n_layers * (max_attn_splits + max_ffn_splits),
            n_layers * 2,
            len(adjncy),
        )

        return xadj, adjncy, eweights, node_types, split_ids, node_weights
