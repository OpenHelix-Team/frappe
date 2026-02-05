import re, sys, os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from peft import get_peft_model, LoraConfig
from timm.models.vision_transformer import RmsNorm, Mlp

from pathlib import Path
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent))
from hub_mixin import CompatiblePyTorchModelHubMixin
from utils import load_encoders
from rdt.flappe_model import RDT
from mid_train_runner import preprocess_raw_image, mean_flat

class MOEExpert(nn.Module,
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
                 device_id=None):
        super(MOEExpert, self).__init__()
        
        self.accelerator = accelerator
        self.learnable_tokens = learnable_tokens
        self.config = config  
        
        if device_id is not None:
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = accelerator.device
        
        if enc_type != None:
            self.encoders, self.encoder_types, self.architectures = load_encoders(enc_type, self.device, resolution)
            self.encoders = [encoder.to(torch.bfloat16) for encoder in self.encoders]
        else:
            raise NotImplementedError()
        target_dims = [encoder.embed_dim for encoder in self.encoders] if enc_type != 'None' else [0]
        
        if learnable_tokens is None:
            raise ValueError("learnable_tokens cannot be None")
        hidden_size = config['rdt']['hidden_size']
        
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
            projector_dim=config['rdt']['hidden_size'],
            learnable_tokens=learnable_tokens,
        )
        
        self.model = self.model.to(self.device)
        
        self.lang_adaptor = self.build_condition_adapter(config['lang_adaptor'],
                                                         in_features=lang_token_dim,
                                                         out_features=hidden_size)
        self.img_adaptor = self.build_condition_adapter(config['img_adaptor'],
                                                        in_features=img_token_dim,
                                                        out_features=hidden_size)
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=state_token_dim * 2,
            out_features=hidden_size)
        
        self.lang_adaptor = self.lang_adaptor.to(self.device)
        self.img_adaptor = self.img_adaptor.to(self.device)
        self.state_adaptor = self.state_adaptor.to(self.device)
        
        self.to(self.device)
        
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
    
    def freeze_rdt_and_adaptors(self):

        for name, param in self.model.named_parameters():
            if 'projectors' not in name and 'learnable_tokens' not in name and 'x_pos_embed' and 'lora' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
            
        for param in self.lang_adaptor.parameters():
            param.requires_grad = False
            if param.device != self.device:
                param.data = param.data.to(self.device)
        for param in self.img_adaptor.parameters():
            param.requires_grad = False
            if param.device != self.device:
                param.data = param.data.to(self.device)
        for param in self.state_adaptor.parameters():
            param.requires_grad = False
            if param.device != self.device:
                param.data = param.data.to(self.device)    
    
    def apply_lora(self, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: List[str]):
       
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        if self.accelerator.is_main_process:
            self.model.print_trainable_parameters()

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
    
    def compute_proj_loss_and_get_hidden_states(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, 
                                               action_gt, action_mask, ctrl_freqs, encoder_depth, future_images):
        """
        Computes the projection loss and returns the hidden states of the last layer.
        Runs a single forward pass to avoid redundant computation.
        Action loss is no longer computed here, as the final loss is now calculated using the fused action.
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        # Sample noise that we'll add to the actions
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        # Sample random diffusion timesteps
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size, ), device=device).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)

        zs = []
        with self.accelerator.autocast():
            for encoder, encoder_type, arch in zip(self.encoders, self.encoder_types, self.architectures):   
                raw_image_ = preprocess_raw_image(future_images, encoder_type)
                if 'theia' in encoder_type:
                    raw_image = raw_image_.permute(0, 2, 3, 1)
                    z = encoder.forward_features(raw_image.to(torch.float32), feature_reduction_method='none')
                else:
                    z = encoder.forward_features(raw_image_)
                    
                if 'dinov2' in encoder_type:  z = z['x_norm_patchtokens']
                
                zs.append(z)

        # Concatenate the state and action tokens to form the input sequence
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # Append the action mask to the input sequence
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        # Align the dimension with the hidden size
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(lang_tokens, img_tokens, state_action_traj)
        # Predict the denoised result with teacher supervision

        pred, zs_future = self.model(state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, lang_mask=lang_attn_mask, encoder_depth=encoder_depth, return_hidden_states=True)

        bsz = zs[0].shape[0]
        proj_loss = 0.0  
        for i, (z, z_tilde) in enumerate(zip(zs, zs_future)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                proj_loss += mean_flat(1-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)

        # return proj_loss and hidden states
        return proj_loss, pred
    
    def get_last_layer_hidden_states_for_prediction(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, 
                                                    action_mask, ctrl_freqs):

        device = next(self.model.parameters()).device
        lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs = \
            lang_tokens.to(device), lang_attn_mask.to(device), img_tokens.to(device), \
            state_tokens.to(device), action_mask.to(device), ctrl_freqs.to(device)

        state_tokens_with_mask = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(lang_tokens, img_tokens, state_tokens_with_mask)
        
        noisy_action = torch.randn(size=(state_tokens.shape[0], self.pred_horizon, self.action_dim),
                                   dtype=state_tokens.dtype,
                                   device=device)
        
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
        with torch.no_grad():
            self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
            
            timesteps_list = list(self.noise_scheduler_sample.timesteps)
            
            for i, t in enumerate(timesteps_list):
                action_traj = torch.cat([noisy_action, action_mask], dim=2)
                action_traj = self.state_adaptor(action_traj)
                state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
                is_last_step = (i == len(timesteps_list) - 1)
                
                if is_last_step:
                    # last step, return hidden states
                    output = self.model(state_action_traj, ctrl_freqs, t.unsqueeze(-1).to(device),
                                       lang_cond, img_cond, lang_mask=lang_attn_mask, 
                                       return_hidden_states=True)
                else:
                    # not last step, update noise
                    model_output, out= self.model(state_action_traj, ctrl_freqs, t.unsqueeze(-1).to(device),
                                             lang_cond, img_cond, lang_mask=lang_attn_mask)
                    
                    noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
                    noisy_action = noisy_action.to(state_tokens.dtype)
            
            noisy_action = noisy_action * action_mask
        
        if isinstance(output, tuple):
            output = output[0]
            
        return output
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_action_loss(*args, **kwargs)

class GateNetwork(nn.Module):
    def __init__(self, hidden_size: int, pred_horizon: int, num_experts: int, dropout: float = 0.1, temperature: float = 1.0):
        super(GateNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.pred_horizon = pred_horizon
        self.num_experts = num_experts
        
        input_dim = self.hidden_size * 3
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_experts),
        )
        
        self.temperature = temperature        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.gate_mlp.modules():
            if isinstance(module, nn.Linear):
                
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, pred_horizon, hidden_size * 3)  
        Returns:
            gate_weights: expert_weights(batch_size, num_experts)
        """
        pooled_output = torch.mean(x, dim=1)  # (batch_size, hidden_size * 3)
        
        logits = self.gate_mlp(pooled_output)  # (batch_size, num_experts)
        gate_weights = F.softmax(logits / self.temperature, dim=-1)
        
        return gate_weights, logits
    
class MOERDTRunner(nn.Module, CompatiblePyTorchModelHubMixin):
    def __init__(self, 
                 expert_configs: List[Dict[str, Any]],
                 action_dim: int,
                 pred_horizon: int,
                 lang_token_dim: int,
                 img_token_dim: int,
                 state_token_dim: int,
                 max_lang_cond_len: int,
                 img_cond_len: int,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16,
                 resolution=256,
                 accelerator=None,
                 gate_hidden_dim: int = 256,
                 use_lora: bool = False,
                 lora_config: Optional[Dict[str, Any]] = None):
        super(MOERDTRunner, self).__init__()
        
        self.accelerator = accelerator
        self.num_experts = len(expert_configs)
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        
        self._expert_configs = expert_configs
        
        self.experts = nn.ModuleList()
        for i, expert_config in enumerate(expert_configs):
            if accelerator.is_main_process:
                print(f"create expert {i+1}: enc_type={expert_config.get('enc_type')}, learnable_tokens={expert_config.get('learnable_tokens')}")

            checkpoint_path = expert_config.get('checkpoint_path')
            if checkpoint_path and os.path.exists(checkpoint_path):
                expert = MOEExpert.from_pretrained(
                    checkpoint_path,
                    enc_type=expert_config.get('enc_type'),
                    resolution=resolution,
                    accelerator=accelerator,
                    learnable_tokens=expert_config.get('learnable_tokens'),
                    action_dim=action_dim,
                    pred_horizon=pred_horizon,
                    config=expert_config['config'],
                    lang_token_dim=lang_token_dim,
                    img_token_dim=img_token_dim,
                    state_token_dim=state_token_dim,
                    max_lang_cond_len=max_lang_cond_len,
                    img_cond_len=img_cond_len,
                    lang_pos_embed_config=lang_pos_embed_config,
                    img_pos_embed_config=img_pos_embed_config,
                    dtype=dtype,
                    device_id=None
                )
                if accelerator.is_main_process:
                    print(f"expert {i+1} checkpoint loaded")
            else:
                if accelerator.is_main_process:
                    print(f"expert {i+1} initialize randomly")
                
                expert = MOEExpert(
                    action_dim=action_dim,
                    pred_horizon=pred_horizon,
                    config=expert_config['config'],
                    lang_token_dim=lang_token_dim,
                    img_token_dim=img_token_dim,
                    state_token_dim=state_token_dim,
                    max_lang_cond_len=max_lang_cond_len,
                    img_cond_len=img_cond_len,
                    lang_pos_embed_config=lang_pos_embed_config,
                    img_pos_embed_config=img_pos_embed_config,
                    dtype=dtype,
                    enc_type=expert_config.get('enc_type'),
                    resolution=resolution,
                    accelerator=accelerator,
                    learnable_tokens=expert_config.get('learnable_tokens'),
                    device_id=None  
                )
            
            if use_lora and lora_config:
                expert.apply_lora(
                    lora_r=lora_config['r'],
                    lora_alpha=lora_config['alpha'],
                    lora_dropout=lora_config['dropout'],
                    target_modules=lora_config['target_modules']
                )
                
            self.experts.append(expert)
        
        self.gate_network = GateNetwork(
            hidden_size=2048, 
            pred_horizon=pred_horizon,
            num_experts=self.num_experts,
            temperature=1.0  
        )
        
        # fusion layer: same as RDT FinalLayer
        class FusionFinalLayer(nn.Module):
            def __init__(self, hidden_size, out_channels):
                super().__init__()
                self.norm_fusion = RmsNorm(hidden_size, eps=1e-6)
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.ffn_fusion = Mlp(in_features=hidden_size,
                                      hidden_features=hidden_size,
                                      out_features=out_channels,
                                      act_layer=approx_gelu,
                                      drop=0)
            
            def forward(self, x):
                x = self.norm_fusion(x)
                x = self.ffn_fusion(x)
                return x
        
        self.fusion_final_layer = FusionFinalLayer(2048, action_dim)
        self._init_fusion_mlp_weights()
        self.gate_network = self.gate_network.to(accelerator.device)
        self.label_smoothing_epsilon = 0.1  

    def _init_fusion_mlp_weights(self):
        
        for module in self.fusion_final_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        final_fc_layer = self.fusion_final_layer.ffn_fusion.fc2
        nn.init.constant_(final_fc_layer.weight, 0)
        nn.init.constant_(final_fc_layer.bias, 0)

    
    def freeze_experts_and_train_moe_and_tokens(self):        
        for expert in self.experts:
            expert.freeze_rdt_and_adaptors()
        
        # ensure gating trainable
        for param in self.gate_network.parameters():
            param.requires_grad = True
        
        # ensure fusion trainable
        for param in self.fusion_final_layer.parameters():
            param.requires_grad = True
    
    def apply_label_smoothing_to_weights(self, gate_weights):
        """
        w_i ← w_i × (1 - ε) + ε/P
        """
        epsilon = self.label_smoothing_epsilon
        P = self.num_experts
        smoothed_weights = gate_weights * (1 - epsilon) + epsilon / P
        return smoothed_weights
    
    def get_gate_input(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs):
        """
        Obtains the gating network input and returns the full hidden states of all experts.
        This function internally performs a full forward pass for each expert.
        """
        
        expert_full_hidden_states = []
        expert_action_hidden_states = []
        
        with torch.no_grad():
            for expert in self.experts:
                
                expert_hidden = expert.get_last_layer_hidden_states_for_prediction(
                    lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs
                )
                expert_full_hidden_states.append(expert_hidden)
                
                # only action part hidden states as input
                action_hidden = expert_hidden[:, :self.pred_horizon, :]
                expert_action_hidden_states.append(action_hidden)

        # (batch_size, pred_horizon, hidden_size * 3)
        gate_input = torch.cat(expert_action_hidden_states, dim=2)
        
        return gate_input, expert_full_hidden_states
    
    def forward_experts(self, expert_inputs, gate_weights):
        expert_proj_losses = []
        expert_hidden_states = []
        
        lang_tokens, lang_attn_mask, img_tokens, state_tokens, \
        action_gt, action_mask, ctrl_freqs, encoder_depth, future_images = expert_inputs

        for i, expert in enumerate(self.experts):
            device = expert.device
            expert_inputs_on_device = [
                lang_tokens.to(device), lang_attn_mask.to(device), img_tokens.to(device), state_tokens.to(device),
                action_gt.to(device), action_mask.to(device), ctrl_freqs.to(device), 
                encoder_depth, future_images.to(device)
            ]
            
            proj_loss, hidden_states = expert.compute_proj_loss_and_get_hidden_states(*expert_inputs_on_device)
            
            expert_hidden_states.append(hidden_states.to(gate_weights.device))
            expert_proj_losses.append(proj_loss)
        
        expert_proj_losses = torch.stack(expert_proj_losses, dim=0)  
        expert_hidden_states = torch.stack(expert_hidden_states, dim=0)  
        
        gate_weights_expanded = gate_weights.unsqueeze(-1).unsqueeze(-1)  
        expert_hidden_states_transposed = expert_hidden_states.transpose(0, 1)  
        
        # weighted sum
        fused_hidden_states = torch.sum(expert_hidden_states_transposed * gate_weights_expanded, dim=1)  

        action_horizon = action_gt.shape[1]
        predicted_hidden_states = fused_hidden_states[:, :action_horizon, :] 

        batch_size, horizon, actual_hidden_size = predicted_hidden_states.shape
        fused_hidden_states_flat = predicted_hidden_states.reshape(-1, actual_hidden_size)  
        
        # final action
        fused_action = self.fusion_final_layer(fused_hidden_states_flat)  
        fused_action = fused_action.reshape(batch_size, horizon, -1) 
        
        expert_proj_losses_expanded = expert_proj_losses.unsqueeze(0).expand(gate_weights.shape[0], -1)  # (batch_size, num_experts)
        combined_proj_loss = torch.sum(expert_proj_losses_expanded, dim=1).mean() / 3.0  # scalar
        
        return combined_proj_loss, expert_proj_losses, fused_action
        
    def compute_logits_penalty(self, logits):
        """
        L_z(x) = (1/S) * Sum_{k=1 to S} (log(Sum_{i=1 to N} e^(g_i^(k))))^2
        S: token number (batch_size)
        N:  (num_experts)
        g: logits (batch_size, num_experts)
        """
        log_sum_exp = torch.logsumexp(logits, dim=1)  
        squared_log_sum_exp = log_sum_exp ** 2  
        penalty = torch.mean(squared_log_sum_exp)
        
        return penalty
    
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, 
                    action_gt, action_mask, ctrl_freqs, encoder_depth, future_images, proj_coeff):
        IGNORE_ACTION_LOSS_SIGNAL = -9999.0
        is_real_action = (action_gt[:, 0, 0] != IGNORE_ACTION_LOSS_SIGNAL)

        gate_input, _ = self.get_gate_input(lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs)
        
        gate_weights, logits = self.gate_network(gate_input)
        
        smoothed_gate_weights = self.apply_label_smoothing_to_weights(gate_weights)
        
        expert_inputs = (lang_tokens, lang_attn_mask, img_tokens, state_tokens, 
                        action_gt, action_mask, ctrl_freqs, encoder_depth, future_images)
        
        combined_proj_loss, expert_proj_losses, fused_action = self.forward_experts(expert_inputs, smoothed_gate_weights)
        
        action_horizon = action_gt.shape[1]
        predicted_actions = fused_action[:, :action_horizon, :]
        
        all_losses = F.mse_loss(predicted_actions, action_gt, reduction='none').mean(dim=list(range(1, len(predicted_actions.shape))))
        masked_losses = all_losses * is_real_action.float()
        
        num_real_actions = is_real_action.sum()
        if num_real_actions > 0:
            final_action_loss = masked_losses.sum() / num_real_actions
        else:
            final_action_loss = torch.tensor(0.0, device=lang_tokens.device)
        
        z_penalty = self.compute_logits_penalty(logits)
        self.z_loss_weight = 0.000001

        total_loss = final_action_loss + combined_proj_loss * proj_coeff + z_penalty * self.z_loss_weight
        
        return total_loss, {
            'final_action_loss': final_action_loss, 
            'combined_proj_loss': combined_proj_loss * proj_coeff,  
            'z_loss': z_penalty * self.z_loss_weight,  
            'gate_weights': smoothed_gate_weights,  
            'original_gate_weights': gate_weights,  
            'expert_proj_losses': expert_proj_losses,  
            'fused_action': fused_action[:, :action_gt.shape[1], :],  
        }

    def forward(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)