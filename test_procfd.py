"""Test: skip_indices + finalize + fd server/client across processes."""
import multiprocessing
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

GRAN = 2 * 1024 * 1024  # 2MB
SOCKET_PATH = "/tmp/vmm_test_fd_server.sock"


def owner_proc(ready_event, done_event):
    import torch
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.empty(1, device="cuda:0")

    from vllm.model_executor.model_loader.vmm_utils import (
        VMMCompositeWeight, start_fd_server)

    w = VMMCompositeWeight(
        device=0,
        sub_component_sizes_bytes=[GRAN, GRAN, GRAN],
        shape=[3 * GRAN // 2],
        dtype=torch.float16,
        exportable=True,
    )
    t = w.tensor
    chunk_elems = GRAN // 2
    t[0:chunk_elems].fill_(1.0)
    t[chunk_elems:2 * chunk_elems].fill_(2.0)
    t[2 * chunk_elems:3 * chunk_elems].fill_(3.0)

    # Export chunk 1 and serve via fd server
    _pid, fd = w.export_chunk(1)
    start_fd_server(SOCKET_PATH, {"2.self_attn.qkv_proj.k_proj": fd})
    print(f"[Owner] Exported chunk 1, fd server running")
    ready_event.set()

    done_event.wait(timeout=30)
    print(f"[Owner] Done, chunk1={t[chunk_elems].item()}")


def consumer_proc(ready_event, done_event):
    import torch
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.empty(1, device="cuda:0")

    from vllm.model_executor.model_loader.vmm_utils import (
        VMMCompositeWeight, request_fd)

    ready_event.wait(timeout=30)
    time.sleep(0.5)

    # Create with hole at index 1
    w = VMMCompositeWeight(
        device=0,
        sub_component_sizes_bytes=[GRAN, GRAN, GRAN],
        shape=[3 * GRAN // 2],
        dtype=torch.float16,
        exportable=False,
        skip_indices={1},
    )

    # Request fd from owner's server
    import_fd = request_fd(SOCKET_PATH, "2.self_attn.qkv_proj.k_proj")
    print(f"[Consumer] Got fd={import_fd} from server")

    w.import_chunk_from_fd(1, import_fd)
    os.close(import_fd)
    w.finalize()

    # Fill owned chunks, check imported chunk
    t = w.tensor
    chunk_elems = GRAN // 2
    t[0:chunk_elems].fill_(5.0)
    t[2 * chunk_elems:3 * chunk_elems].fill_(7.0)

    q_val = t[0].item()
    k_val = t[chunk_elems].item()
    v_val = t[2 * chunk_elems].item()
    print(f"[Consumer] Q={q_val}, K={k_val}, V={v_val}")
    assert q_val == 5.0, f"Q should be 5.0, got {q_val}"
    assert k_val == 2.0, f"K should be 2.0 (from owner), got {k_val}"
    assert v_val == 7.0, f"V should be 7.0, got {v_val}"
    print("[Consumer] PASS: skip_indices + fd_server + finalize works")
    done_event.set()


if __name__ == "__main__":
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    done = ctx.Event()

    p_owner = ctx.Process(target=owner_proc, args=(ready, done))
    p_consumer = ctx.Process(target=consumer_proc, args=(ready, done))

    p_owner.start()
    p_consumer.start()
    p_consumer.join(timeout=60)
    p_owner.join(timeout=60)

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    if p_owner.exitcode != 0 or p_consumer.exitcode != 0:
        print(f"FAILED: owner={p_owner.exitcode}, consumer={p_consumer.exitcode}")
        exit(1)
    print("=== skip_indices + fd_server + finalize: PASSED ===")
