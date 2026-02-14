# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from typing import Optional, cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    fastsafetensors_weights_iterator, filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference, maybe_download_from_modelscope,
    multi_thread_pt_weights_iterator,
    multi_thread_safetensors_weights_iterator, np_cache_weights_iterator,
    pt_weights_iterator, safetensors_weights_iterator)
from vllm.platforms import current_platform

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary, cudaIpcMemHandle_t
from vllm.model_executor.model_loader.vmm_utils import (
    VMMCompositeWeight, start_fd_server, request_fd)
import ctypes
import cupy as cp
import base64
import json
import re

logger = init_logger(__name__)


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: Optional[str]
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: Optional[list[str]] = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(f"Unexpected extra config keys for load format "
                             f"{load_config.load_format}: "
                             f"{unexpected_keys}")

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: Optional[str],
        fall_back_to_pt: bool,
        allow_patterns_overrides: Optional[list[str]],
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = (maybe_download_from_modelscope(
            model_name_or_path, revision) or model_name_or_path)

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # Some quantized models use .pt files for storing the weights.
        if load_format == "auto":
            allow_patterns = ["*.safetensors", "*.bin"]
        elif (load_format == "safetensors"
              or load_format == "fastsafetensors"):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == "mistral":
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == "pt":
            allow_patterns = ["*.pt"]
        elif load_format == "npcache":
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file)
        else:
            hf_weights_files = filter_files_not_needed_for_inference(
                hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
            self, source: "Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        extra_config = self.load_config.model_loader_extra_config
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt,
            source.allow_patterns_overrides)
        if self.load_config.load_format == "npcache":
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        elif use_safetensors:
            if self.load_config.load_format == "fastsafetensors":
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            else:
                if extra_config.get("enable_multithread_load"):
                    weights_iterator = (
                        multi_thread_safetensors_weights_iterator(
                            hf_weights_files,
                            self.load_config.use_tqdm_on_load,
                            max_workers=extra_config.get(
                                "num_threads", self.DEFAULT_NUM_THREADS),
                        ))
                else:
                    weights_iterator = safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        self.load_config.safetensors_load_strategy,
                    )
        else:
            if extra_config.get("enable_multithread_load"):
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                    max_workers=extra_config.get("num_threads",
                                                 self.DEFAULT_NUM_THREADS),
                )
            else:
                weights_iterator = pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                )

        if current_platform.is_tpu():
            from vllm.platforms.tpu import USE_TPU_COMMONS

            if not USE_TPU_COMMONS:
                # In PyTorch XLA, we should call `torch_xla.sync`
                # frequently so that not too many ops are accumulated
                # in the XLA program.
                import torch_xla

                def _xla_weights_iterator(iterator: Generator):
                    for weights in iterator:
                        yield weights
                        torch_xla.sync(wait=False)

                weights_iterator = _xla_weights_iterator(weights_iterator)

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor)
                for (name, tensor) in weights_iterator)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load",
                                    True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides",
                                             None),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model,
                              model_config.revision,
                              fall_back_to_pt=True,
                              allow_patterns_overrides=None)

    def _setup_vmm_sharing(self, model: nn.Module, model_config: ModelConfig
                           ) -> tuple[set[str], dict]:
        """Set up VMM-based partial weight sharing.

        Reads the sharing spec, pre-allocates VMM tensors, and for consumers
        imports shared chunks from the owner via Unix socket.

        The spec file uses shard_order to map checkpoint names to indices,
        so no model-specific knowledge is needed.

        Returns:
            (skip_names, vmm_weights) where:
            - skip_names: set of checkpoint weight name substrings to filter
            - vmm_weights: dict mapping "layer.module" to VMMCompositeWeight
        """
        spec_path = model_config.shared_layers_spec_path
        with open(spec_path) as f:
            spec = json.load(f)

        model_id = os.environ.get("VMM_MODEL_ID", "")
        is_owner = (model_id == spec["owner"])
        socket_path = spec.get("socket_path", "/tmp/vmm_sharing.sock")

        module_map = dict(model.named_modules())
        skip_names: set[str] = set()
        vmm_weights: dict[str, VMMCompositeWeight] = {}
        fd_map: dict[str, int] = {}  # for owner's fd server

        for entry in spec["components"]:
            layer = entry["layer"]
            module_name = entry["module"]       # e.g., "self_attn.qkv_proj"
            shard_order = entry["shard_order"]  # e.g., ["q_proj", "k_proj", "v_proj"]
            shared_shards = entry["shared"]     # e.g., ["k_proj"]

            # Map shared names to shard indices via shard_order
            shared_indices = set()
            for shard_name in shared_shards:
                idx = shard_order.index(shard_name)
                shared_indices.add(idx)

            # Find the module
            full_module_name = f"model.layers.{layer}.{module_name}"
            module = module_map.get(full_module_name)
            if module is None:
                raise ValueError(f"Module {full_module_name} not found")

            # Get sub-component sizes from module
            output_sizes = module.output_sizes  # e.g., [4096, 4096, 4096]
            weight_shape = list(module.weight.shape)  # e.g., [12288, 4096]
            input_size = weight_shape[1]
            dtype = module.weight.dtype

            sub_sizes_bytes = [s * input_size * dtype.itemsize
                               for s in output_sizes]

            # Create VMM tensor
            if is_owner:
                vmm_w = VMMCompositeWeight(
                    device=0,
                    sub_component_sizes_bytes=sub_sizes_bytes,
                    shape=weight_shape,
                    dtype=dtype,
                    exportable=True,
                )
            else:
                vmm_w = VMMCompositeWeight(
                    device=0,
                    sub_component_sizes_bytes=sub_sizes_bytes,
                    shape=weight_shape,
                    dtype=dtype,
                    exportable=False,
                    skip_indices=shared_indices,
                )
                # Import shared chunks from owner
                for shard_name in shared_shards:
                    idx = shard_order.index(shard_name)
                    key = f"{layer}.{module_name}.{shard_name}"
                    fd = request_fd(socket_path, key)
                    vmm_w.import_chunk_from_fd(idx, fd)
                    os.close(fd)
                vmm_w.finalize()

            # Swap module weight data (preserve ModelWeightParameter + weight_loader)
            with torch.no_grad():
                module.weight.data = vmm_w.tensor

            vmm_weights[f"{layer}.{module_name}"] = vmm_w

            # Build skip set for consumer: skip shared checkpoint weights
            # e.g., for module "self_attn.qkv_proj", shard "k_proj" at layer 2,
            # skip weights matching "model.layers.2.self_attn.k_proj"
            if not is_owner:
                module_prefix = module_name.rsplit(".", 1)[0]  # "self_attn"
                for shard_name in shared_shards:
                    skip_names.add(
                        f"model.layers.{layer}.{module_prefix}.{shard_name}")

            logger.info("VMM %s: layer %d %s, shared=%s, indices=%s",
                        "owner" if is_owner else "consumer",
                        layer, module_name, shared_shards, shared_indices)

        # Owner: export shared chunks and start fd server
        if is_owner:
            for entry in spec["components"]:
                layer = entry["layer"]
                module_name = entry["module"]
                shard_order = entry["shard_order"]
                vmm_w = vmm_weights[f"{layer}.{module_name}"]

                for shard_name in entry["shared"]:
                    idx = shard_order.index(shard_name)
                    _pid, fd = vmm_w.export_chunk(idx)
                    key = f"{layer}.{module_name}.{shard_name}"
                    fd_map[key] = fd

            start_fd_server(socket_path, fd_map)

        return skip_names, vmm_weights

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        weights_to_load = {name for name, _ in model.named_parameters()}
        logger.info(f"Weights to load: {weights_to_load}")

        # Set up VMM sharing if spec exists
        skip_names: set[str] = set()
        vmm_weights: dict = {}
        if (model_config.shared_layers_spec_path
                and os.path.exists(model_config.shared_layers_spec_path)):
            spec_path = model_config.shared_layers_spec_path
            # Check if it's a VMM spec (JSON with "components" key) vs old handles.jsonl
            with open(spec_path) as f:
                first_char = f.read(1)
            if first_char == '{':
                # VMM sharing spec
                skip_names, vmm_weights = self._setup_vmm_sharing(
                    model, model_config)

        # Get weight iterator, optionally filtered
        weights_iter = self.get_all_weights(model_config, model)
        if skip_names:
            original_iter = weights_iter
            def filtered_weights(iterator, skip):
                for name, tensor in iterator:
                    if any(s in name for s in skip):
                        logger.info("VMM: skipping weight %s", name)
                        continue
                    yield name, tensor
            weights_iter = filtered_weights(original_iter, skip_names)

        loaded_weights = model.load_weights(weights_iter)

        # Existing IPC sharing (whole-module) — only if NOT using VMM
        if not vmm_weights:
            lib = CudaRTLibrary()
            lib.cudaSetDevice(0)
            tmp = lib.cudaMalloc(1)
            lib.cudaFree(tmp)

            if (model_config.shared_layers_spec_path
                    and os.path.exists(model_config.shared_layers_spec_path)):
                # Check if it's old-style handles.jsonl
                with open(model_config.shared_layers_spec_path) as f:
                    first_char = f.read(1)
                if first_char != '{':
                    self.load_weight_pointers(lib, model, model_config)

        self.counter_after_loading_weights = time.perf_counter()
        logger.info(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights -
            self.counter_before_loading_weights)
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking currently.
        if model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError("Following weights were not initialized from "
                                 f"checkpoint: {weights_not_loaded}")

    def store_weight_pointers(self, lib, model: nn.Module, model_config: ModelConfig) -> None:
        model_name = model_config.model

        for param_name, param in model.named_parameters():
            # Parse layer + component
            # Example: model.layers.1.self_attn.qkv_proj.weight
            print(param_name)
            m = re.match(r"model\.layers\.(\d+)\.(.+)\.weight", param_name)
            if not m:
                # raise ValueError(f"Unrecognized parameter format: {param_name}")
                continue

            layer_idx = int(m.group(1))
            component = m.group(2)

            ALLOWED_COMPONENTS = (
                "self_attn.qkv_proj",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "mlp.gate_proj",
                "mlp.gate_up_proj",
            )

            if component not in ALLOWED_COMPONENTS:
                continue

            handle = lib.cudaIpcGetMemHandle(param.data_ptr())
            handle_bytes = ctypes.string_at(ctypes.addressof(handle), 128)
            handle_b64 = base64.b64encode(handle_bytes).decode("ascii")

            record = {
                "model_name": model_name,
                "layer": layer_idx,
                "component": component,
                "handle": handle_b64,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
            }

            with open(model_config.shared_layers_ptrs_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def load_weight_pointers(self, lib, model: nn.Module, model_config: ModelConfig) -> None:
        module_map = dict(model.named_modules())

        with open(model_config.shared_layers_ptrs_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                layer = rec["layer"]
                component = rec["component"]
                shape = torch.Size(rec["shape"])
                dtype = getattr(torch, rec["dtype"].split(".")[-1])
                handle_bytes = base64.b64decode(rec["handle"])

                module_name = f"model.layers.{layer}.{component}"
                module = module_map[module_name]

                self.load_ipc_param_into_module(lib, module, handle_bytes, shape, dtype)

    @torch._dynamo.disable
    def load_ipc_param_into_module(self, lib, module, handle_bytes, shape, dtype, device="cuda:0"):
        """Replace a module's parameter with a tensor backed by a CUDA IPC pointer."""

        lib = CudaRTLibrary()
        lib.cudaSetDevice(0)
        # ---- 1. Decode handle ----
        handle = cudaIpcMemHandle_t.from_buffer_copy(handle_bytes)
        pointer = lib.cudaIpcOpenMemHandle(handle)
        if pointer is None or pointer.value == 0:
            raise RuntimeError("cudaIpcOpenMemHandle failed")

        # ---- 2. Build CuPy array from raw pointer ----
        cp_dtype = {
            torch.float16: cp.float16,
            torch.bfloat16: cp.uint16,
            torch.float32: cp.float32,
            torch.int8: cp.int8,
        }[dtype]

        nbytes = cp.dtype(cp_dtype).itemsize * int(cp.prod(cp.asarray(shape)))

        mem = cp.cuda.UnownedMemory(pointer.value, nbytes, owner=None)
        ptr = cp.cuda.MemoryPointer(mem, 0)
        cupy_arr = cp.ndarray(shape, dtype=cp_dtype, memptr=ptr)

        # ---- 3. Convert to Torch tensor *outside* Dynamo ----
        torch_arr = torch.as_tensor(cupy_arr, device=device)
        if dtype == torch.bfloat16:
            torch_arr = torch_arr.view(torch.bfloat16)

        # ---- 4. Replace the parameter on the module ----
        with torch.no_grad():
            setattr(module, "weight", torch.nn.Parameter(torch_arr))