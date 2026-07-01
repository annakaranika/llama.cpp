import os
import json
import numpy as np
import matplotlib
# Headless/script default, but don't clobber a notebook's inline backend.
if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import Device, AttentionSplitStrategy, FFNSplitStrategy
from .model import ModelSpec
from .network import Network
from .policies.single_node import simulate_single_node
from .policies.pipeline import simulate_pipeline
from .policies.tensor import simulate_tensor_parallel
from .policies.metis import simulate_metis
from .policies.alpa import simulate_alpa
from .policies.hybrid_pp_tp import simulate_hybrid_pp_tp
from .policies.msct import simulate_msct

# Define a custom ModelSpec wrapper for Decode Phase
class DecodeModelSpec(ModelSpec):
    def __init__(self, original_model, context_len):
        # Copy attributes from original model
        self.__dict__.update(original_model.__dict__)
        self.context_len = context_len

    def layer_compute_flops(self, seq_len=1, dev_q_heads=None, dev_kv_heads=None, ffn_fraction=1.0):
        # seq_len is 1 for a decode step. Projections process 1 token.
        # Attention scores matrix multiplication scales with the history context_len.
        q_heads = dev_q_heads if dev_q_heads is not None else self.num_heads
        kv_heads = dev_kv_heads if dev_kv_heads is not None else self.num_kv_heads
        
        q_proj = 1 * self.d_model * (q_heads * self.d_k)
        kv_proj = 2 * 1 * self.d_model * (kv_heads * self.d_k)
        out_proj = 1 * self.d_model * (q_heads * self.d_k)
        
        # Attention score math scales with context_len
        q_fraction = q_heads / self.num_heads
        attn_compute = self.context_len * self.d_model * q_fraction
        
        # FFN scales with 1 token
        ffn = self.ffn_compute_flops(1, ffn_fraction=ffn_fraction)
        
        return q_proj + kv_proj + attn_compute + out_proj + ffn

    def attn_compute_flops(self, seq_len, dev_q_heads=None, dev_kv_heads=None):
        q_heads = dev_q_heads if dev_q_heads is not None else self.num_heads
        kv_heads = dev_kv_heads if dev_kv_heads is not None else self.num_kv_heads
        
        q_proj = seq_len * self.d_model * (q_heads * self.d_k)
        kv_proj = 2 * seq_len * self.d_model * (kv_heads * self.d_k)
        
        q_fraction = q_heads / self.num_heads
        attn_compute = self.context_len * self.d_model * q_fraction
        
        out_proj = seq_len * self.d_model * (q_heads * self.d_k)
        return q_proj + kv_proj + attn_compute + out_proj

    def attn_memory_gb_per_kv_head(self, seq_len=512):
        # Memory usage must use the context_len history for KV cache
        return super().attn_memory_gb_per_kv_head(self.context_len)

    def memory_usage_per_device(self, seq_len=512, *args, **kwargs):
        # Redirect memory calculations to context_len
        return super().memory_usage_per_device(self.context_len, *args, **kwargs)


def run_prefill_simulation(policy_func, model, net, prompt_len, **kwargs):
    """
    Simulate the prefill (TTFT) phase for a prompt of length prompt_len.

    Pipeline and SCT policies now correctly scale their inter-stage activation
    transfer by seq_len internally, so no bandwidth manipulation is needed here.
    We simply call the policy with seq_len=prompt_len and gen_tokens=1.
    """
    res = policy_func(model, net, seq_len=prompt_len, gen_tokens=1, **kwargs)
    return res["total_latency_s"]


def run_decode_simulation(policy_func, model, net, prompt_len, **kwargs):
    """
    Simulate a 1-token decode step with context history prompt_len.
    """
    decode_model = DecodeModelSpec(model, prompt_len)
    res = policy_func(decode_model, net, seq_len=1, gen_tokens=1, **kwargs)
    return res["total_latency_s"]


def run_latency_evaluation(net, model, prompt_lengths=None, output_dir=None):
    """
    Run TTFT and TPT evaluation across all policies and generate plots.
    """
    if prompt_lengths is None:
        prompt_lengths = [128, 256, 512]
    if output_dir is None:
        output_dir = "sim_output/latency"
    os.makedirs(output_dir, exist_ok=True)
    
    policies = {
        'single_node': simulate_single_node,
        'pipeline': simulate_pipeline,
        'tensor': simulate_tensor_parallel,
        'metis': simulate_metis,
        'alpa': simulate_alpa,
        'hybrid_pp_tp': simulate_hybrid_pp_tp,
        'msct': simulate_msct
    }

    policy_labels = {
        'single_node': 'Single Node',
        'pipeline': 'Pipeline',
        'tensor': 'Tensor Parallel',
        'metis': 'METIS',
        'alpa': 'Alpa',
        'hybrid_pp_tp': 'Hybrid PP+TP',
        'msct': 'm-SCT'
    }

    ttft_results = {p: [] for p in policies}
    tpt_results = {p: [] for p in policies}
    
    print("=" * 70)
    print("Running Prefill (TTFT) and Decode (TPT) Simulations")
    print("=" * 70)
    
    for prompt_len in prompt_lengths:
        print(f"\n--- Prompt Length: {prompt_len} tokens ---")
        for pname, policy_func in policies.items():
            try:
                # 1. Prefill (TTFT)
                ttft = run_prefill_simulation(policy_func, model, net, prompt_len)
                # 2. Decode (TPT)
                tpt = run_decode_simulation(policy_func, model, net, prompt_len)
                
                ttft_results[pname].append(ttft)
                tpt_results[pname].append(tpt)
                print(f"  {policy_labels[pname]:<18} | TTFT: {ttft:.4f}s | TPT: {tpt:.4f}s")
            except Exception as e:
                print(f"  {policy_labels[pname]:<18} | Failed: {e}")
                ttft_results[pname].append(float('nan'))
                tpt_results[pname].append(float('nan'))

    # Save raw JSON results
    raw_data = {
        "prompt_lengths": prompt_lengths,
        "ttft": ttft_results,
        "tpt": tpt_results
    }
    with open(os.path.join(output_dir, "latency_results.json"), "w") as f:
        json.dump(raw_data, f, indent=2)

    # 1. Plot TTFT vs Prompt Length
    plt.figure(figsize=(8, 5))
    for pname in policies:
        plt.plot(prompt_lengths, ttft_results[pname], marker='o', label=policy_labels[pname])
    plt.xlabel('Prompt Length (tokens)', fontweight='bold')
    plt.ylabel('Time to First Token (TTFT, seconds)', fontweight='bold')
    plt.title('TTFT vs. Prompt Length across Policies', fontweight='bold', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ttft_vs_prompt_len.pdf"))
    plt.savefig(os.path.join(output_dir, "ttft_vs_prompt_len.png"), dpi=300)
    plt.close()

    # 2. Plot TPT vs Prompt Length
    plt.figure(figsize=(8, 5))
    for pname in policies:
        plt.plot(prompt_lengths, tpt_results[pname], marker='s', label=policy_labels[pname])
    plt.xlabel('Prompt Length (tokens)', fontweight='bold')
    plt.ylabel('Time Per Token (TPT, seconds/token)', fontweight='bold')
    plt.title('TPT (Decode Step) vs. Prompt Length', fontweight='bold', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tpt_vs_prompt_len.pdf"))
    plt.savefig(os.path.join(output_dir, "tpt_vs_prompt_len.png"), dpi=300)
    plt.close()

    # 3. Plot Cumulative Latency over generated tokens (Context = 512)
    plt.figure(figsize=(8, 5))
    gen_tokens_range = np.arange(0, 128 + 1, 8)
    for pname in policies:
        if len(ttft_results[pname]) > 2:
            ttft_512 = ttft_results[pname][2]
            tpt_512 = tpt_results[pname][2]
            if np.isnan(ttft_512) or np.isnan(tpt_512):
                continue
            cum_latencies = ttft_512 + gen_tokens_range * tpt_512
            plt.plot(gen_tokens_range, cum_latencies, label=policy_labels[pname])
    plt.xlabel('Number of Generated Tokens', fontweight='bold')
    plt.ylabel('Cumulative Latency (seconds)', fontweight='bold')
    plt.title('Cumulative Latency vs. Tokens Generated (Context = 512)', fontweight='bold', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_latency.pdf"))
    plt.savefig(os.path.join(output_dir, "cumulative_latency.png"), dpi=300)
    plt.close()
    
    print("\n" + "=" * 70)
    print(f"Latency plots saved successfully under: {output_dir}")
    print("=" * 70)
    
    return raw_data
