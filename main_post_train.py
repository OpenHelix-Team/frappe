import argparse
import os
from accelerate.logging import get_logger
from train.post_train import train 

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
        help="The type of encoder to use.",
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

if __name__ == "__main__":
    logger = get_logger(__name__)
    args = parse_args()
    train(args, logger)