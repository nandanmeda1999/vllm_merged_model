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

from vllm.distributed.device_communicators.cuda_wrapper import CudaDriverLibrary, CudaRTLibrary, cudaIpcMemHandle_t
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

    def _setup_vmm_sharing(self, model: nn.Module, model_config: ModelConfig):
        spec_path = model_config.shared_layers_spec_path
        handles_path = model_config.shared_layers_ptrs_path
        model_id = os.path.basename(os.path.normpath(model_config.model))

        drv = CudaDriverLibrary()
        my_physical_device = drv.get_physical_device()

        with open(spec_path) as f:
            spec = json.load(f)

        handles = {}
        if os.path.exists(handles_path):
            with open(handles_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    for model_name, info in entry.items():
                        if info.get("physical_device") == my_physical_device:
                            handles[model_name] = info

        module_map = dict(model.named_modules())
        skip_names: set[str] = set()
        vmm_weights: dict[str, VMMCompositeWeight] = {}
        my_export_components = []
        fd_map: dict[str, int] = {}
        socket_path = os.environ.get("VMM_SOCKET_PATH", "")
        if socket_path == "":
            raise ValueError("VMM_SOCKET_PATH environment variable is not set")
        if not os.path.exists(socket_path):
            raise ValueError(f"Socket path {socket_path} does not exist")
        socket_path = os.path.join(socket_path, f"vmm_{os.getpid()}.sock")

        def resolve_component(component):
            """ Map high-level component to module/shard info"""

            if component.startswith("self_attn.") and component.split(".")[1] in (
                "q_proj", "k_proj", "v_proj"
            ):
                shard = component.split(".")[1]
                return "self_attn.qkv_proj", ["q_proj", "k_proj", "v_proj"], [shard]

            if component.startswith("mlp.") and component.split(".")[1] in (
                "gate_proj", "up_proj"
            ):
                shard = component.split(".")[1]
                return "mlp.gate_up_proj", ["gate_proj", "up_proj"], [shard]

            if component == "self_attn.o_proj":
                return "self_attn.o_proj", ["o_proj"], ["o_proj"]

            if component == "mlp.down_proj":
                return "mlp.down_proj", ["down_proj"], ["down_proj"]

            raise ValueError(f"Unknown component {component}")
        

        def model_exports_component(handles, model_name, layer,
                             module_name, shared_shards):
            """Check if a model has exported the required component."""

            info = handles.get(model_name)
            if not info:
                return False

            for comp in info.get("components", []):
                if comp["layer"] != layer:
                    continue
                if comp["module"] != module_name:
                    continue

                if any(s in comp.get("shared", []) for s in shared_shards):
                    return True

            return False

        for group in spec:
            if len(group) == 1:
                continue

            mine = next((e for e in group if e["model"] == model_id), None)
            if not mine:
                continue

            layer = mine["layer"]
            component = mine["component"]

            module_name, shard_order, shared_shards = resolve_component(component)

            shared_indices = set()
            for shard_name in shared_shards:
                idx = shard_order.index(shard_name)
                shared_indices.add(idx)

            full_module_name = f"model.layers.{layer}.{module_name}"
            module = module_map.get(full_module_name)
            if module is None:
                raise ValueError(f"Module {full_module_name} not found")
                        
            weight_shape = list(module.weight.shape)
            input_size = weight_shape[1]
            if hasattr(module, "output_sizes"):
                # fused projections like qkv_proj, gate_up_proj
                output_sizes = module.output_sizes
            else:
                # single projection layer
                output_sizes = [weight_shape[0]]
            dtype = module.weight.dtype

            sub_sizes_bytes = [s * input_size * dtype.itemsize
                               for s in output_sizes]

            owner_socket = None
            owner_model = None

            for other in group:
                if other["model"] == model_id:
                    continue

                other_model = other["model"]
                other_layer = other["layer"]
                other_component = other["component"]

                other_module_name, other_shard_order, other_shards = resolve_component(other_component)

                assert len(other_shards) == 1, "Currently only support one shared shard per component"

                if model_exports_component(handles, other_model, other_layer, other_module_name, other_shards):
                    owner_socket = handles[other_model]["socket"]
                    owner_model = other_model
                    break

            vmm_weights_key = f"{layer}.{module_name}"

            # Consumer path
            if owner_socket:
                if vmm_weights_key in vmm_weights:
                    vmm_w = vmm_weights[vmm_weights_key]
                    logger.info("Reusing existing VMM weight for layer %d %s ", layer, module_name)
                else:
                    vmm_w = VMMCompositeWeight(
                        device=0,
                        sub_component_sizes_bytes=sub_sizes_bytes,
                        shape=weight_shape,
                        dtype=dtype,
                        exportable=True,
                        skip_indices=shared_indices,
                    )

                for shard_name in shared_shards:
                    idx = shard_order.index(shard_name)
                    key = f"{other_layer}.{other_module_name}.{other_shards[0]}"
                    fd = request_fd(owner_socket, key)
                    vmm_w.import_chunk_from_fd(idx, fd)
                    os.close(fd)
                vmm_w.finalize() 

                skip_names.add(
                    f"model.layers.{layer}.{component}"
                )

            # Owner path
            else:
                if vmm_weights_key in vmm_weights:
                    vmm_w = vmm_weights[vmm_weights_key]
                    logger.info("Reusing existing VMM weight for layer %d %s", layer, module_name)
                else:
                    vmm_w = VMMCompositeWeight(
                        device=0,
                        sub_component_sizes_bytes=sub_sizes_bytes,
                        shape=weight_shape,
                        dtype=dtype,
                        exportable=True,
                    )
                
                for shard in shared_shards:
                    idx = shard_order.index(shard)
                    _, fd = vmm_w.export_chunk(idx)
                    key = f"{layer}.{module_name}.{shard}"
                    fd_map[key] = fd

                my_export_components.append({
                    "layer": layer,
                    "module": module_name,
                    "shard_order": shard_order,
                    "shared": shared_shards,
                })

            with torch.no_grad():
                module.weight.data = vmm_w.tensor

            torch.cuda.empty_cache()

            vmm_weights[vmm_weights_key] = vmm_w

            logger.info("VMM %s: layer %d %s, shared=%s, indices=%s",
                        "consumer" if owner_socket else "owner",
                        layer, module_name, shared_shards, shared_indices)

        if fd_map:
            start_fd_server(socket_path, fd_map)

            with open(handles_path, "a") as f:
                json.dump({
                    model_id: {
                        "socket": socket_path,
                        "physical_device": my_physical_device,
                        "components": my_export_components,
                    }
                }, f)
                f.write("\n")

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

        # torch.cuda.empty_cache()
        loaded_weights = model.load_weights(weights_iter)

        self.counter_after_loading_weights = time.perf_counter()
        logger.info(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights -
            self.counter_before_loading_weights)
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking currently.
        if model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            filtered_not_loaded = set()

            for w in weights_not_loaded:
                # Extract layer prefix. e.g. model.layers.2.self_attn
                prefix, name, param = w.rsplit(".", 2)

                if name in {"o_proj", "down_proj"}:
                    if f"{prefix}.{name}" not in skip_names:
                        filtered_not_loaded.add(w)
                    continue

                if name == "qkv_proj":
                    shards = [
                        f"{prefix}.q_proj",
                        f"{prefix}.k_proj",
                        f"{prefix}.v_proj",
                    ]
                    if not all(s in skip_names for s in shards):
                        filtered_not_loaded.add(w)
                    continue

                if name == "gate_up_proj":
                    shards = [
                        f"{prefix}.gate_proj",
                        f"{prefix}.up_proj",
                    ]
                    if not all(s in skip_names for s in shards):
                        filtered_not_loaded.add(w)
                    continue

                filtered_not_loaded.add(w)

            if filtered_not_loaded:
                raise ValueError("Following weights were not initialized from "
                                 f"checkpoint: {filtered_not_loaded}")