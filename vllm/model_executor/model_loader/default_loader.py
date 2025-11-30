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
import ctypes
import cupy as cp
import base64
import json

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

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        weights_to_load = {name for name, _ in model.named_parameters()}
        model_to_copy_from = model_config.model_to_copy_from

        loaded_weights = model.load_weights(
            self.get_all_weights(model_config, model))

        if model_to_copy_from is not None:
            # print("========", model_to_copy_from.model.layers[5])
            # model.model.layers[1].self_attn = model_to_copy_from.model.layers[1].self_attn
            model.model.layers[1].mlp = model_to_copy_from.model.layers[1].mlp
            model.model.layers[2].mlp = model_to_copy_from.model.layers[2].mlp
        
        # param_to_copy = None
        # for name, param in model.named_parameters():
        #     print(name, hex(param.data_ptr()))
        #     param_to_copy = param
        #     break
        
        lib = CudaRTLibrary()
        lib.cudaSetDevice(0)
        tmp = lib.cudaMalloc(1)
        lib.cudaFree(tmp)  # ensure context exists

        shared_mem_file = "cuda_ipc_handles.json"
        if not os.path.exists(shared_mem_file):
            params_to_copy = [
                ("model.layers.0.self_attn.qkv_proj.weight", model.model.layers[0].self_attn.qkv_proj.weight),
                ("model.layers.1.self_attn.qkv_proj.weight", model.model.layers[1].self_attn.qkv_proj.weight),
                # 'model.layers.1.self_attn.qkv_proj.weight', 'model.layers.1.self_attn.o_proj.weight', 'model.layers.1.mlp.gate_up_proj.weight', 
                # 'model.layers.1.mlp.down_proj.weight', 'model.layers.1.input_layernorm.weight', 'model.layers.1.post_attention_layernorm.weight'
            ]
            export_list = []
            for name, param in params_to_copy:
                handle = lib.cudaIpcGetMemHandle(param.data_ptr())
                handle_bytes = ctypes.string_at(ctypes.addressof(handle), size=128)
                handle_b64 = base64.b64encode(handle_bytes).decode("ascii")
                entry = {
                    "name": name,
                    "handle": handle_b64,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype) 
                }
                export_list.append(entry)
            with open(shared_mem_file, "w") as f:
                json.dump(export_list, f, indent=2)
        else:
            with open(shared_mem_file, "r") as f:
                records = json.load(f)
            module_map = dict(model.named_modules())
            for rec in records:
                name = rec["name"]
                module_name, param_name = name.rsplit(".", 1)
                module = module_map[module_name]
                shape = torch.Size(rec["shape"])
                dtype = getattr(torch, rec["dtype"].split(".")[-1])
                handle_b64 = rec["handle"]
                handle_bytes = base64.b64decode(handle_b64)
                self.load_ipc_param_into_module(module, handle_bytes, shape, dtype)

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


    @torch._dynamo.disable   # <-- critical
    def load_ipc_param_into_module(self, module, handle_bytes, shape, dtype, device="cuda:0"):
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