import copy  
import logging
import math
import os
import datetime
timeout_seconds = datetime.timedelta(hours=3).total_seconds()
os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = str(int(timeout_seconds))

import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DeepSpeedPlugin, ProjectConfiguration
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm

current_file = Path(__file__)
sys.path.append(str(current_file.parent))

from models.post_train_runner import MOERDTRunner


from models.ema_model import EMAModel 
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from train.sample import log_sample_res

if is_wandb_available():
    import wandb

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train MOE RDT model")
    
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_text_encoder_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vision_encoder_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--CONFIG_NAME", type=str, required=True)
    
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--dataset_type", type=str, default="finetune")
    parser.add_argument("--load_from_hdf5", action="store_true")
    parser.add_argument("--image_aug", action="store_true")
    parser.add_argument("--cond_mask_prob", type=float, default=0.1)
    parser.add_argument("--cam_ext_mask_prob", type=float, default=0.1)
    parser.add_argument("--state_noise_snr", type=float, default=0.1)
    parser.add_argument("--precomp_lang_embed", action="store_true")
    
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[256, 512],
        default=256,
        required=False,
        help="The resolution of the encoder.",
    )
    parser.add_argument(
        "--proj_coeff",
        type=float,
        default=0.01,
        help="The coefficient for the projection loss.",
    )
    parser.add_argument(
        "--encoder_depth",
        type=int,
        default=8,
        required=False,
        help="The depth of the encoder.",
    )
    parser.add_argument(
        "--wm_horizon",
        type=int,
        default=8,
        required=False,
        help="The number of the future step to use for the future image.",
    )
    parser.add_argument(
        "--learnable_tokens",
        type=int,
        default=64,
        required=False,
        help="The number of learnable tokens to use.",
    )
    parser.add_argument(
        "--train_only_learnable_tokens",
        action="store_true",
        help="If set, freezes the entire RDT model and adaptors, then only trains the learnable tokens."
    )
    parser.add_argument(
        "--enc_type",
        type=str,
        default=None,
        required=False,
        help="The type of encoder to use. Choose between ['None', 'dinov2', 'mocov3'].",
    )
    
    
    #LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--checkpointing_period", type=int, default=1000)
    parser.add_argument("--sample_period", type=int, default=500)
    parser.add_argument("--num_sample_batches", type=int, default=2, help="Number of batches to sample during evaluation")
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=2.0)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    
    return parser.parse_args()

def load_expert_configs(config_path: str, args, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    expert_configs = []
    for expert_name, expert_config in config['experts'].items():
        expert_config_dict = {
            'name': expert_config['name'],
            'config': base_config['model'],  
            'learnable_tokens': expert_config.get('learnable_tokens', args.learnable_tokens),
            'enc_type': expert_config.get('enc_type', args.enc_type),
            'checkpoint_path': expert_config.get('checkpoint_path', None) 
        }
        expert_configs.append(expert_config_dict)
    
    return expert_configs

def main():
    args = parse_args()
    
    if args.seed is not None:
        set_seed(args.seed)
    
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    
    with open(args.model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    args.output_dir = model_config["checkpoint_path"]
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=(DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.precomp_lang_embed:
        tokenizer, text_encoder = None, None
    else:
        text_embedder = T5Embedder(
            from_pretrained=args.pretrained_text_encoder_name_or_path,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=accelerator.device,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    
    vision_encoder = SiglipVisionTower(vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    image_processor = vision_encoder.image_processor
    

    expert_configs = load_expert_configs(args.model_config_path, args, config)
    
    img_cond_len = (config["common"]["img_history_size"] * config["common"]["num_cameras"] *
                    vision_encoder.num_patches)
    
    lora_config_dict = None
    if args.use_lora:
        logger.info("train as LoRA setting")
        
        target_modules = [
            "qkv",        
            "q",          
            "kv",         
            "proj",       
            "fc1",        
            "fc2",        
        ]
        
        lora_config_dict = {
            'r': args.lora_r,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
            'target_modules': target_modules
        }

    model_class = MOERDTRunner

    model = model_class(
        expert_configs=expert_configs,
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        lang_token_dim=config["model"]["lang_token_dim"],
        img_token_dim=config["model"]["img_token_dim"],
        state_token_dim=config["model"]["state_token_dim"],
        max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
        img_cond_len=img_cond_len,
        lang_pos_embed_config=[
            ("lang", -config["dataset"]["tokenizer_max_length"]),
        ],
        img_pos_embed_config=[
            (
                "image",
                (
                    config["common"]["img_history_size"],
                    config["common"]["num_cameras"],
                    -vision_encoder.num_patches,
                ),
            ),
        ],
        dtype=weight_dtype,
        resolution=args.resolution,
        accelerator=accelerator,
        use_lora=args.use_lora,
        lora_config=lora_config_dict
    )
    
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model
                if isinstance(model_to_save, type(accelerator.unwrap_model(model))):
                    model_to_save.save_pretrained(output_dir)
    
    accelerator.register_save_state_pre_hook(save_model_hook)
    
    if args.gradient_checkpointing:
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size *
                              accelerator.num_processes)
    
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.train_only_learnable_tokens:
         if isinstance(accelerator.unwrap_model(model), MOERDTRunner):
            unwrapped_model = accelerator.unwrap_model(model)
            for expert in unwrapped_model.experts:
                expert.freeze_rdt_and_adaptors()
            for param in unwrapped_model.gate_network.parameters():
                param.requires_grad = False
            gate_network_frozen = True
    else:
        if isinstance(accelerator.unwrap_model(model), MOERDTRunner):
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.freeze_experts_and_train_moe_and_tokens()

    trainable_params = []
    if not args.train_only_learnable_tokens:
        trainable_params.extend([p for p in model.gate_network.parameters() if p.requires_grad])
        trainable_params.extend([p for p in model.fusion_final_layer.parameters() if p.requires_grad])
    for expert in model.experts:
        if hasattr(expert.model, 'learnable_tokens') and expert.model.learnable_tokens is not None:
            if expert.model.learnable_tokens.requires_grad:
                trainable_params.append(expert.model.learnable_tokens)
        if hasattr(expert.model, 'projectors') and expert.model.projectors is not None:
            for projector in expert.model.projectors:
                trainable_params.extend([p for p in projector.parameters() if p.requires_grad])
        for name, param in expert.model.named_parameters():
            if 'x_pos_embed' in name and param.requires_grad:
                trainable_params.append(param)
        if args.use_lora:
            for name, param in expert.model.named_parameters():
                if 'lora' in name and param.requires_grad:
                    trainable_params.append(param)
                
    
    optimizer = optimizer_class(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    train_dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
        wm_horizon=args.wm_horizon,
    )
    
    sample_dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
        wm_horizon=args.wm_horizon,
    )
    
    data_collator = DataCollatorForVLAConsumerDataset(tokenizer)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    model, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, sample_dataloader, lr_scheduler
    )
    
    
    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    if vision_encoder is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "VLA_MOE",
            config=vars(args),
            init_kwargs={"wandb": {
                "name": f"RoboTwin_MOE_RDT_{args.CONFIG_NAME}",
            }},
        )
    
    if (args.resume_from_checkpoint is None and args.pretrained_model_name_or_path is not None
            and os.path.isfile(args.pretrained_model_name_or_path)):
        logger.info("Loading from a pretrained checkpoint.")
        checkpoint = torch.load(args.pretrained_model_name_or_path)
        model.module.load_state_dict(checkpoint["module"])
    
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path))
            except:
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(
                    os.path.join(
                        args.output_dir,
                        path,
                        "pytorch_model",
                        "mp_rank_00_model_states.pt",
                    ))
                model.module.load_state_dict(checkpoint["module"])
            
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    

        gate_params_ids = {id(p) for p in unwrapped_model.gate_network.parameters()}
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['params'] = [
                p for p in optimizer.param_groups[0]['params'] if id(p) not in gate_params_ids
            ]
            logger.info("Removed gate network parameters from the optimizer upon resuming.")
        else:
            logger.warning("Optimizer has multiple param groups, which is not handled by the gate network freezing logic.")
        gate_network_frozen = True

    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    loss_for_log = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)
        
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                images = batch["images"].to(dtype=weight_dtype)
                states = batch["states"].to(dtype=weight_dtype)
                states = states[:, -1:, :]  
                actions = batch["actions"].to(dtype=weight_dtype)
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                ctrl_freqs = batch["ctrl_freqs"]
                
                with torch.no_grad():
                    batch_size, _, C, H, W = images.shape
                    input_images = images[:, :6]  
                    future_images = images[:, 6:7]  
                    future_images = F.interpolate(
                        future_images.squeeze(1), 
                        size=(args.resolution, args.resolution),  
                        mode='bicubic',
                        align_corners=False  
                    )
                    image_embeds = vision_encoder(input_images.reshape(-1, C, H, W)).detach()
                    image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))
                    
                    lang_attn_mask = batch["lang_attn_mask"]
                    text_embeds = (batch["lang_embeds"].to(dtype=weight_dtype) 
                                 if args.precomp_lang_embed 
                                 else text_encoder(input_ids=batch["input_ids"], 
                                                 attention_mask=lang_attn_mask)["last_hidden_state"].detach())
                
                state_elem_mask = state_elem_mask.unsqueeze(1)

                total_loss, loss_dict = model(
                    lang_tokens=text_embeds,
                    lang_attn_mask=lang_attn_mask,
                    img_tokens=image_embeds,
                    state_tokens=states,
                    action_gt=actions,
                    action_mask=state_elem_mask,
                    ctrl_freqs=ctrl_freqs,
                    encoder_depth=args.encoder_depth,
                    future_images=future_images,
                    proj_coeff=args.proj_coeff,
                )
                
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    ema_save_path = os.path.join(save_path, f"ema")
                    logger.info(f"Saved state to {save_path}")
                
                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    sample_loss_for_log = log_sample_res(
                        text_encoder,
                        vision_encoder,
                        model,
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataset.get_dataset_id2name(),
                        sample_dataloader,
                        logger,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)
            
            logs = {
                "loss": total_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "final_action_loss": loss_dict['final_action_loss'].item(),
                "combined_proj_loss": loss_dict['combined_proj_loss'].item(),
                "z_loss":loss_dict['z_loss'].item(),
            }
            
            gate_weights = loss_dict['gate_weights']
            for i in range(gate_weights.shape[1]):
                logs[f"expert_{i}_usage"] = gate_weights[:, i].mean().item()
            
            if 'original_gate_weights' in loss_dict:
                original_weights = loss_dict['original_gate_weights']
                for i in range(original_weights.shape[1]):
                    logs[f"original_expert_{i}_usage"] = original_weights[:, i].mean().item()

            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                logs["peak_mem_mb"] = f"{peak_memory_mb:.2f}"
            
            progress_bar.set_postfix(
                loss=f"{logs['loss']:.4f}",
                action_loss=f"{logs['final_action_loss']:.4f}",
                proj_loss=f"{logs['combined_proj_loss']:.4f}",
                lr=f"{logs['lr']:.2e}",
                mem_mb=f"{logs.get('peak_mem_mb', 'N/A')}"                
            )
            
            expert_usage_str = " ".join([f"E{i}:{logs[f'expert_{i}_usage']:.3f}" for i in range(gate_weights.shape[1])])
            print(f"\nExpert Usage: {expert_usage_str}")
            
            if global_step % 100 == 0:
                if 'original_gate_weights' in loss_dict:
                    original_weights = loss_dict['original_gate_weights']
                    original_usage = [original_weights[:, i].mean().item() for i in range(original_weights.shape[1])]
                    original_str = " ".join([f"E{i}:{usage:.3f}" for i, usage in enumerate(original_usage)])
                    print(f"Original Weights: {original_str}")
                
                smoothed_usage = [gate_weights[:, i].mean().item() for i in range(gate_weights.shape[1])]
                smoothed_str = " ".join([f"E{i}:{usage:.3f}" for i, usage in enumerate(smoothed_usage)])
                print(f"Smoothed Weights: {smoothed_str}")
                print(f"Label Smoothing ε: {model.label_smoothing_epsilon}")
                print("-" * 50)
            
            logs.update(loss_for_log)
            accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        
        logger.info(f"Saved Model to {args.output_dir}")
        
        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
            )
    
    accelerator.end_training()

if __name__ == "__main__":
    main()