import os
from pathlib import Path
from typing import Dict, Optional, Union
import math
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.constants import (PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE)
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, is_torch_available
import torch
import torch.nn.functional as F

from peft import PeftModel

class CompatiblePyTorchModelHubMixin(PyTorchModelHubMixin):
    """Mixin class to load Pytorch models from the Hub."""

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        # To bypass saving into safetensor by default
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        torch.save(model_to_save.state_dict(), save_directory / PYTORCH_WEIGHTS_NAME)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        enc_type: str = None,
        resolution: int = 256,
        accelerator: Optional[any] = None, 
        learnable_tokens: Optional[any] = None,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        if cls.__name__ == 'MOEExpert':
            model = cls(enc_type=enc_type, resolution=resolution, accelerator=accelerator, learnable_tokens=learnable_tokens, **model_kwargs)
        elif cls.__name__ == 'RDTRunner':
            model = cls(enc_type=enc_type, resolution=resolution, accelerator=accelerator, learnable_tokens=learnable_tokens, **model_kwargs)
        else:
            model = cls(resolution=resolution, accelerator=accelerator,  **model_kwargs)       
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            try:
                model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
                return cls._load_as_pickle(model, model_file, map_location, strict)
            except FileNotFoundError:
                model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
                return cls._load_as_safetensor(model, model_file, map_location, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_pickle(model, model_file, map_location, strict)

    @staticmethod
    def _load_and_interpolate(model, state_dict):
        model_state_dict = model.state_dict()
        keys_to_pop = []
        POS_EMBED_PREFIX_LEN = 35

        for key in list(state_dict.keys()):
            if key in model_state_dict:
                checkpoint_shape = state_dict[key].shape
                model_shape = model_state_dict[key].shape

                if checkpoint_shape != model_shape:
                    
                    if 'pos_embed' in key:
                        # print(f"INFO: Shape mismatch for '{key}'. Applying 2D interpolation with a fixed prefix of {POS_EMBED_PREFIX_LEN}.")
                        # print(f"  - Checkpoint shape: {checkpoint_shape}")
                        # print(f"  - Model shape:      {model_shape}")
                        pos_embed_checkpoint = state_dict[key]

                        if checkpoint_shape[1] < POS_EMBED_PREFIX_LEN or model_shape[1] < POS_EMBED_PREFIX_LEN:
                            # print(f"  ✗ Skipping '{key}'. Token length is smaller than the prefix length {POS_EMBED_PREFIX_LEN}.")
                            keys_to_pop.append(key)
                            continue

                        prefix_embed = pos_embed_checkpoint[:, :POS_EMBED_PREFIX_LEN, :]
                        patch_embed_checkpoint = pos_embed_checkpoint[:, POS_EMBED_PREFIX_LEN:, :]
                        num_patches_model = model_shape[1] - POS_EMBED_PREFIX_LEN
                        
                        if patch_embed_checkpoint.shape[1] <= 0 or num_patches_model <= 0:
                            #  print(f"  ✗ Skipping '{key}'. Invalid number of patch tokens after removing prefix.")
                             keys_to_pop.append(key)
                             continue

                        grid_size_checkpoint = int(math.sqrt(patch_embed_checkpoint.shape[1]))
                        grid_size_model = int(math.sqrt(num_patches_model))

                        if grid_size_checkpoint**2 != patch_embed_checkpoint.shape[1] or grid_size_model**2 != num_patches_model:
                            # print(f"  ✗ Skipping '{key}'. Number of patch tokens ({patch_embed_checkpoint.shape[1]} or {num_patches_model}) is not a perfect square.")
                            keys_to_pop.append(key)
                            continue
                        
                        # Reshape to 2D grid and interpolate
                        embedding_dim = patch_embed_checkpoint.shape[-1]
                        patch_embed_2d = patch_embed_checkpoint.reshape(1, grid_size_checkpoint, grid_size_checkpoint, embedding_dim).permute(0, 3, 1, 2)
                        patch_embed_interpolated = F.interpolate(patch_embed_2d, size=(grid_size_model, grid_size_model), mode='bilinear', align_corners=False)
                        
                        # Reshape back and concatenate with the fixed prefix
                        patch_embed_interpolated = patch_embed_interpolated.permute(0, 2, 3, 1).reshape(1, num_patches_model, embedding_dim)
                        resized_tensor = torch.cat((prefix_embed, patch_embed_interpolated), dim=1)
                        
                        if resized_tensor.shape == model_shape:
                            # print(f"  ✓ Successfully resized '{key}' to {resized_tensor.shape}")
                            state_dict[key] = resized_tensor
                        else:
                            # print(f"  ✗ Failed to resize '{key}'. Final shape {resized_tensor.shape} != model shape {model_shape}.")
                            keys_to_pop.append(key)

                    elif 'learnable_tokens' in key:
                        # print(f"INFO: Shape mismatch for '{key}'. Applying 2D interpolation for all tokens.")
                        # print(f"  - Checkpoint shape: {checkpoint_shape}")
                        # print(f"  - Model shape:      {model_shape}")

                        patch_embed_checkpoint = state_dict[key]
                        num_patches_model = model_shape[1]
                        
                        grid_size_checkpoint = int(math.sqrt(patch_embed_checkpoint.shape[1]))
                        grid_size_model = int(math.sqrt(num_patches_model))

                        if grid_size_checkpoint**2 != patch_embed_checkpoint.shape[1] or grid_size_model**2 != num_patches_model:
                            # print(f"  ✗ Skipping '{key}'. Number of tokens is not a perfect square.")
                            keys_to_pop.append(key)
                            continue

                        # Reshape, interpolate, and reshape back
                        embedding_dim = patch_embed_checkpoint.shape[-1]
                        patch_embed_2d = patch_embed_checkpoint.reshape(1, grid_size_checkpoint, grid_size_checkpoint, embedding_dim).permute(0, 3, 1, 2)
                        patch_embed_interpolated = F.interpolate(patch_embed_2d, size=(grid_size_model, grid_size_model), mode='bilinear', align_corners=False)
                        resized_tensor = patch_embed_interpolated.permute(0, 2, 3, 1).reshape(1, num_patches_model, embedding_dim)
                        
                        if resized_tensor.shape == model_shape:
                            # print(f"  ✓ Successfully resized '{key}' to {resized_tensor.shape}")
                            state_dict[key] = resized_tensor
                        else:
                            # print(f"  ✗ Failed to resize '{key}'. Final shape {resized_tensor.shape} != model shape {model_shape}.")
                            keys_to_pop.append(key)
                    else:
                        keys_to_pop.append(key)
        for key in keys_to_pop:
            # print(f"Skipping loading '{key}' due to shape mismatch:")
            # print(f"  - Checkpoint shape: {state_dict.get(key, 'Already removed').shape if state_dict.get(key) is not None else 'Already removed'}")
            # print(f"  - Model shape:      {model_state_dict.get(key, 'Not in model').shape if model_state_dict.get(key) is not None else 'Not in model'}")
            if key in state_dict:
                state_dict.pop(key)
        
        
        loaded_special_tokens = []
        for key in state_dict.keys():
            if 'learnable_tokens' in key or 'pos_embed' in key:
                # print(f"✓ '{key}' loaded with shape: {state_dict[key].shape}")
                loaded_special_tokens.append(key)
        
        model_special_tokens = [k for k,v in model.named_parameters() if 'learnable_tokens' in k or 'pos_embed' in k]
        # for token_name in model_special_tokens:
            # if token_name not in loaded_special_tokens and token_name not in [k for k in keys_to_pop]:
                # print(f"✗ {token_name} was not found or loaded from the checkpoint.")
        
        # print("\n--- Checking LoRA Loading Status ---")
        
        expected_lora_keys = {name for name, _ in model.named_parameters() if 'lora_' in name}
        provided_lora_keys = {key for key in state_dict.keys() if 'lora_' in key}
        keys_that_will_be_loaded = expected_lora_keys.intersection(provided_lora_keys)
        keys_missing_from_checkpoint = expected_lora_keys - provided_lora_keys

        # if not expected_lora_keys:
        #      print("INFO: Model architecture does not contain LoRA parameters. Skipping check.")
        # elif not keys_that_will_be_loaded:
        #     print("✗ LORA LOADING FAILED: No LoRA parameters found in the checkpoint that match the model's architecture.")
        # elif keys_missing_from_checkpoint:
        #     print("⚠️ LORA LOADING WARNING: Partially loaded.")
        #     print(f"  ✓ Will load {len(keys_that_will_be_loaded)} LoRA parameters.")
        #     print(f"  ✗ Could not find {len(keys_missing_from_checkpoint)} expected LoRA parameters in the checkpoint. Examples:")
        #     for key in list(keys_missing_from_checkpoint)[:5]:
        #         print(f"    - {key}")
        # else:
        #     print(f"✓ LORA LOADING SUCCESS: Found all {len(keys_that_will_be_loaded)} expected LoRA parameters in the checkpoint.")

        # print("--- LoRA Check Complete ---")

        model.load_state_dict(state_dict, strict=False)
        return model
        
    @staticmethod
    def _load_as_pickle(model, model_file, map_location, strict):
        state_dict = torch.load(model_file, map_location=map_location)
        return CompatiblePyTorchModelHubMixin._load_and_interpolate(model, state_dict)

    @staticmethod
    def _load_as_safetensor(model, model_file, map_location, strict):
        from safetensors.torch import load_file
        state_dict = load_file(model_file, device=map_location)
        return CompatiblePyTorchModelHubMixin._load_and_interpolate(model, state_dict)
