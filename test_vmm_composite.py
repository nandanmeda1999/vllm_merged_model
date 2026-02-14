"""Test 2: VMMCompositeWeight — multi-chunk contiguous tensor."""

import torch
# Ensure CUDA context exists
torch.cuda.init()
torch.cuda.set_device(0)
_ = torch.empty(1, device="cuda:0")

from vllm.model_executor.model_loader.vmm_utils import VMMCompositeWeight

print("=== Test 2: VMMCompositeWeight ===\n")

# Simulate QKV for a model with:
#   hidden_size = 4096, num_heads = 32, head_dim = 128
#   Q: 4096 rows, K: 4096 rows, V: 4096 rows
#   dtype: bfloat16 (2 bytes per element)
hidden_size = 4096
q_rows = 4096   # 32 heads * 128 head_dim
k_rows = 4096   # 32 kv_heads * 128 head_dim
v_rows = 4096

dtype = torch.bfloat16
bytes_per_element = 2

q_bytes = q_rows * hidden_size * bytes_per_element  # 32 MB
k_bytes = k_rows * hidden_size * bytes_per_element  # 32 MB
v_bytes = v_rows * hidden_size * bytes_per_element  # 32 MB

total_rows = q_rows + k_rows + v_rows  # 12288

print(f"Sub-component sizes: Q={q_bytes/1024/1024:.0f}MB, "
      f"K={k_bytes/1024/1024:.0f}MB, V={v_bytes/1024/1024:.0f}MB")

# 1. Create VMMCompositeWeight
w = VMMCompositeWeight(
    device=0,
    sub_component_sizes_bytes=[q_bytes, k_bytes, v_bytes],
    shape=[total_rows, hidden_size],
    dtype=dtype,
    exportable=True,
)

t = w.tensor
print(f"[OK] Created VMM tensor: shape={t.shape}, dtype={t.dtype}, "
      f"device={t.device}, contiguous={t.is_contiguous()}")

assert t.shape == torch.Size([12288, 4096])
assert t.dtype == torch.bfloat16
assert t.is_contiguous()

# 2. Write different values to each sub-component region
t[0:q_rows, :].fill_(1.0)           # Q region
t[q_rows:q_rows+k_rows, :].fill_(2.0)  # K region
t[q_rows+k_rows:, :].fill_(3.0)     # V region

# 3. Verify values
assert t[0, 0].item() == 1.0, f"Q region: expected 1.0, got {t[0, 0].item()}"
assert t[q_rows, 0].item() == 2.0, f"K region: expected 2.0, got {t[q_rows, 0].item()}"
assert t[q_rows + k_rows, 0].item() == 3.0, f"V region: expected 3.0, got {t[q_rows+k_rows, 0].item()}"
print(f"[OK] Write/read per-region works: Q=1.0, K=2.0, V=3.0")

# 4. Verify it works as a GEMM operand (simulating the forward pass)
batch_size = 4
x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda:0")
output = torch.nn.functional.linear(x, t)  # x @ t.T
assert output.shape == (batch_size, total_rows)
q_out, k_out, v_out = output.split([q_rows, k_rows, v_rows], dim=-1)
print(f"[OK] GEMM works: input={x.shape}, weight={t.shape}, "
      f"output={output.shape}")
print(f"     Q_out={q_out.shape}, K_out={k_out.shape}, V_out={v_out.shape}")

# 5. Test export
pid, fd = w.export_chunk(1)  # Export K chunk
print(f"[OK] Exported K chunk: pid={pid}, fd={fd}")

# 6. Verify tensor data is still correct after export
assert t[q_rows, 0].item() == 2.0
print(f"[OK] Data intact after export")

# 7. Test split semantics (mimicking LlamaAttention.forward)
q_size = q_rows
kv_size = k_rows
qkv_output = torch.nn.functional.linear(x, t)
q, k, v = qkv_output.split([q_size, kv_size, kv_size], dim=-1)
assert q.shape == (batch_size, q_size)
assert k.shape == (batch_size, kv_size)
assert v.shape == (batch_size, kv_size)
print(f"[OK] QKV split works: q={q.shape}, k={k.shape}, v={v.shape}")

print("\n=== All VMMCompositeWeight tests passed! ===")
