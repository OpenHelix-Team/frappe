import re, sys, os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from pathlib import Path
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent))
from hub_mixin import CompatiblePyTorchModelHubMixin
from rdt.flappe_model import RDT
from utils import load_encoders
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from peft import get_peft_model, LoraConfig
from typing import List 

MAIN_VISION_ENCODER_MEAN = [0.5, 0.5, 0.5]
MAIN_VISION_ENCODER_STD = [0.5, 0.5, 0.5]
CLIP_DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_DEFAULT_STD = [0.26862954,0.26130258, 0.27577711]

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def preprocess_raw_image(x, enc_type):
    mean = torch.tensor(MAIN_VISION_ENCODER_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(MAIN_VISION_ENCODER_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    
    x_denormalized = x * std + mean

    x_raw = x_denormalized * 255.0
    
    x = torch.clamp(x_raw, 0, 255)

    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        #keep consistent with the official version
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov2' in enc_type:
        x = x / 255.
        #keep consistent with the official version
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x) 
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'theia' in enc_type:
        return x
    elif 'vit' in enc_type:
        x = x / 255.
        #keep consistent with the official version
        x = Normalize(MAIN_VISION_ENCODER_MEAN, MAIN_VISION_ENCODER_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x

class RDTRunner(nn.Module,
                CompatiblePyTorchModelHubMixin):

    def __init__(self,
                 *,
                 action_dim,
                 pred_horizon,
                 config,
                 lang_token_dim,
                 img_token_dim,
                 state_token_dim,
                 max_lang_cond_len,
                 img_cond_len,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16,
                 enc_type=None,
                 resolution=256,
                 accelerator=None,
                 learnable_tokens=None,
                 use_lora: bool = False,
                 lora_config: Optional[Dict[str, Any]] = None):
        super(RDTRunner, self).__init__()

        self.accelerator = accelerator
        self.learnable_tokens = learnable_tokens
        hidden_size = 2048
        if enc_type != None:
            self.encoders, self.encoder_types, self.architectures = load_encoders(enc_type, accelerator.device, resolution)
            self.encoders = [encoder.to(torch.bfloat16) for encoder in self.encoders]

            # Freeze the encoders to prevent gradient calculation and ensure they are not trained.
            for encoder in self.encoders:
                encoder.requires_grad_(False)
                encoder.eval()
        else:
            raise NotImplementedError()
        target_dims = [encoder.embed_dim for encoder in self.encoders] if enc_type != 'None' else [0]


        self.model = RDT(
            output_dim=action_dim,
            horizon=32,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
            z_dims=target_dims,
            projector_dim=2048,
            learnable_tokens=learnable_tokens,
        )

        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(config['lang_adaptor'],
                                                         in_features=lang_token_dim,
                                                         out_features=hidden_size)
        self.img_adaptor = self.build_condition_adapter(config['img_adaptor'],
                                                        in_features=img_token_dim,
                                                        out_features=hidden_size)
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=state_token_dim * 2,  # state + state mask (indicator)
            out_features=hidden_size)

        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)

        return adpated_lang, adpated_img, adpated_state

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs):
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(size=(state_traj.shape[0], self.pred_horizon, self.action_dim),
                                   dtype=dtype,
                                   device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)

        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)

            # Predict the model output
            model_output = self.model(state_action_traj,
                                      ctrl_freqs,
                                      t.unsqueeze(-1).to(device),
                                      lang_cond,
                                      img_cond,
                                      lang_mask=lang_attn_mask)

            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)

        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action


    def apply_lora(self, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: List[str]):
        """
        Applies LoRA to the RDT model.
        """
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("LoRA has been applied to the RDT model.")
        self.model.print_trainable_parameters()
    
    # ========= Inference  ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim),
            which should be a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(lang_tokens, img_tokens, state_tokens)

        # Run sampling
        action_pred = self.conditional_sample(
            lang_cond,
            lang_attn_mask,
            img_cond,
            state_traj,
            action_mask,
            ctrl_freqs,
        )

        return action_pred

    # ========= Train  ============
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)

    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_gt, action_mask,
                    ctrl_freqs, encoder_depth, future_images) -> torch.Tensor:
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        # if only compute the align loss, we keep the data format same while give the action data special notions
        IGNORE_ACTION_LOSS_SIGNAL = -9999.0

        is_real_action = (action_gt[:, 0, 0] != IGNORE_ACTION_LOSS_SIGNAL)

        # Sample noise that we'll add to the actions
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        # Sample random diffusion timesteps
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size, ), device=device).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)

        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(lang_tokens, img_tokens, state_action_traj)
        
        zs = []
        with self.accelerator.autocast():
            for encoder, encoder_type, arch in zip(self.encoders, self.encoder_types, self.architectures):   
                raw_image_ = preprocess_raw_image(future_images, encoder_type)
                if 'theia' in encoder_type:
                    raw_image = raw_image_.permute(0, 2, 3, 1)
                    input_for_theia = raw_image.to(torch.uint8)
                    z = encoder.forward_features(input_for_theia.to(torch.float32), feature_reduction_method='none')
                else:
                    z = encoder.forward_features(raw_image_)
                    
                if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']

                zs.append(z)

        pred, zs_future = self.model(state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, lang_mask=lang_attn_mask, encoder_depth=encoder_depth)

        pred_type = self.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        all_losses = F.mse_loss(pred, target, reduction='none').mean(dim=list(range(1, len(pred.shape))))
        # if action is invalid, mask out the loss
        masked_losses = all_losses * is_real_action.float()

        num_real_actions = is_real_action.sum()
        if num_real_actions > 0:
            loss = masked_losses.sum() / num_real_actions
        else:
            loss = torch.tensor(0.0, device=device)
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_future)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(1-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        return loss, proj_loss
