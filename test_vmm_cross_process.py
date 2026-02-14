"""Test 3: Cross-process VMM sharing — two processes share the K chunk.

Uses Unix domain socket with SCM_RIGHTS to properly pass CUDA DMA-BUF
file descriptors between processes.

Process A (owner):
  - Allocates QKV via VMM (3 physical chunks)
  - Fills Q=1.0, K=2.0, V=3.0
  - Exports K chunk, sends fd to Process B via Unix socket
  - Waits for Process B to signal done, then verifies own data

Process B (consumer):
  - Allocates QKV via VMM (3 physical chunks)
  - Fills Q=5.0, K=6.0, V=7.0
  - Receives K fd from Process A via Unix socket
  - Imports K (replacing its own K chunk)
  - Verifies K=2.0 (from Process A), Q=5.0, V=7.0 (own)
"""

import multiprocessing
import os
import socket
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

SOCKET_PATH = "/tmp/vmm_test_socket"


def run_owner(ready_event, done_event):
    """Process A: owner of K chunk."""
    import torch
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.empty(1, device="cuda:0")

    from vllm.model_executor.model_loader.vmm_utils import VMMCompositeWeight, send_fd

    hidden = 4096
    q_rows, k_rows, v_rows = 4096, 4096, 4096
    bpe = 2  # bfloat16

    w = VMMCompositeWeight(
        device=0,
        sub_component_sizes_bytes=[q_rows * hidden * bpe,
                                   k_rows * hidden * bpe,
                                   v_rows * hidden * bpe],
        shape=[q_rows + k_rows + v_rows, hidden],
        dtype=torch.bfloat16,
        exportable=True,
    )

    t = w.tensor
    t[0:q_rows, :].fill_(1.0)
    t[q_rows:q_rows + k_rows, :].fill_(2.0)
    t[q_rows + k_rows:, :].fill_(3.0)

    # Export K chunk
    pid, fd = w.export_chunk(1)  # K is index 1
    print(f"[Owner] QKV allocated, K exported: pid={pid}, fd={fd}")

    # Set up Unix socket server and send the fd
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)

    # Signal that we're ready
    ready_event.set()

    # Accept connection from consumer and send fd
    conn, _ = server.accept()
    send_fd(conn, fd)
    print(f"[Owner] Sent K fd={fd} to consumer")
    conn.close()
    server.close()

    # Wait for consumer to finish
    done_event.wait(timeout=30)

    # Verify owner's data wasn't corrupted
    assert t[0, 0].item() == 1.0, f"Owner Q corrupted: {t[0,0].item()}"
    assert t[q_rows, 0].item() == 2.0, f"Owner K corrupted: {t[q_rows,0].item()}"
    assert t[q_rows + k_rows, 0].item() == 3.0, f"Owner V corrupted"
    print(f"[Owner] PASS: Data integrity verified after consumer ran")


def run_consumer(ready_event, done_event):
    """Process B: imports K from owner."""
    import time
    import torch
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.empty(1, device="cuda:0")

    from vllm.model_executor.model_loader.vmm_utils import VMMCompositeWeight, recv_fd

    # Wait for owner to be ready
    ready_event.wait(timeout=30)
    time.sleep(0.5)  # Let socket bind complete

    hidden = 4096
    q_rows, k_rows, v_rows = 4096, 4096, 4096
    bpe = 2

    w = VMMCompositeWeight(
        device=0,
        sub_component_sizes_bytes=[q_rows * hidden * bpe,
                                   k_rows * hidden * bpe,
                                   v_rows * hidden * bpe],
        shape=[q_rows + k_rows + v_rows, hidden],
        dtype=torch.bfloat16,
        exportable=False,
    )

    t = w.tensor
    t[0:q_rows, :].fill_(5.0)
    t[q_rows:q_rows + k_rows, :].fill_(6.0)
    t[q_rows + k_rows:, :].fill_(7.0)
    print(f"[Consumer] Before import: Q={t[0,0].item()}, K={t[q_rows,0].item()}, V={t[q_rows+k_rows,0].item()}")

    # Connect to owner and receive fd
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(SOCKET_PATH)
    received_fd = recv_fd(client)
    client.close()
    print(f"[Consumer] Received K fd={received_fd} from owner")

    # Import K chunk using the received fd
    w.import_chunk_from_fd(1, received_fd)
    os.close(received_fd)

    # Verify: K should be 2.0 (from owner), Q and V should be ours
    q_val = t[0, 0].item()
    k_val = t[q_rows, 0].item()
    v_val = t[q_rows + k_rows, 0].item()
    print(f"[Consumer] After import:  Q={q_val}, K={k_val}, V={v_val}")

    assert q_val == 5.0, f"Consumer Q should be 5.0, got {q_val}"
    assert k_val == 2.0, f"Consumer K should be 2.0 (from owner), got {k_val}"
    assert v_val == 7.0, f"Consumer V should be 7.0, got {v_val}"
    print(f"[Consumer] PASS: K=2.0 (shared), Q=5.0 and V=7.0 (own)")

    # Verify GEMM still works with composite tensor
    x = torch.randn(4, hidden, dtype=torch.bfloat16, device="cuda:0")
    output = torch.nn.functional.linear(x, t)
    q_out, k_out, v_out = output.split([q_rows, k_rows, v_rows], dim=-1)
    assert output.shape == (4, q_rows + k_rows + v_rows)
    print(f"[Consumer] PASS: GEMM works with shared K: {output.shape}")

    done_event.set()


def main():
    print("=== Test 3: Cross-Process VMM Sharing ===\n")

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    ctx = multiprocessing.get_context("spawn")
    ready_event = ctx.Event()
    done_event = ctx.Event()

    owner = ctx.Process(target=run_owner, args=(ready_event, done_event))
    consumer = ctx.Process(target=run_consumer, args=(ready_event, done_event))

    owner.start()
    consumer.start()

    consumer.join(timeout=60)
    owner.join(timeout=60)

    # Cleanup
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    if owner.exitcode != 0:
        print(f"\nFAILED: Owner process exited with code {owner.exitcode}")
        sys.exit(1)
    if consumer.exitcode != 0:
        print(f"\nFAILED: Consumer process exited with code {consumer.exitcode}")
        sys.exit(1)

    print("\n=== Cross-process VMM sharing test passed! ===")


if __name__ == "__main__":
    main()
