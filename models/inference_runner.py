import re, sys, os
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from peft import get_peft_model, LoraConfig,PeftModel 
from timm.models.vision_transformer import RmsNorm, Mlp

from pathlib import Path
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent))
from hub_mixin import CompatiblePyTorchModelHubMixin
from mid_train_runner import preprocess_raw_image, mean_flat
from rdt.flappe_model import RDT



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

        if enc_type == "dinov2-vit-b":
            target_dims =[768]
        elif enc_type == "clip-vit-h":
            target_dims =[1280]
        elif enc_type == "vit-huge-patch":
            target_dims =[1280]

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
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
    
        self.fixedtimesteps = self.noise_scheduler_sample.timesteps      
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

    def apply_lora(self, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: List[str]):
       
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )

        self.model = get_peft_model(self.model, lora_config)

    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)

        return adpated_lang, adpated_img, adpated_state
    
    def get_last_layer_hidden_states_for_prediction(self, lang_cond, img_cond, state_traj, 
                                                    action_mask, ctrl_freqs, 
                                                    noisy_action=None):
        device = lang_cond.device
        dtype = lang_cond.dtype
        
        if noisy_action is None:
            noisy_action = torch.randn(size=(state_traj.shape[0], self.pred_horizon, self.action_dim),
                                    dtype=dtype, device=device)
        action_mask_expanded = action_mask.expand(-1, self.pred_horizon, -1)
        self.fixedtimesteps = self.fixedtimesteps.to(device)  
        with torch.no_grad():
            self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)    
            for i, t in enumerate(self.fixedtimesteps):
                action_traj_input = torch.cat([noisy_action, action_mask_expanded], dim=2)
                action_traj = self.state_adaptor(action_traj_input)
                state_action_traj = torch.cat([state_traj, action_traj], dim=1)
                t_input = t.unsqueeze(-1).to(device)
                model_output, out = self.model(state_action_traj, ctrl_freqs, t_input,
                                            lang_cond, img_cond, eval=True)
                    
                    
                noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample

        if isinstance(out, tuple):
            hidden_states = out[0] 
        else:
            hidden_states = out
                
        return hidden_states
    
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
            
            if use_lora and lora_config:
                expert.apply_lora(
                    lora_r=lora_config['r'],
                    lora_alpha=lora_config['alpha'],
                    lora_dropout=lora_config['dropout'],
                    target_modules=lora_config['target_modules']
                )
            expert.eval()    
            self.experts.append(expert)

        self.gate_network = GateNetwork(
            hidden_size=2048, 
            pred_horizon=pred_horizon,
            num_experts=self.num_experts,
            temperature=1.0  
        )

        self.streams = [torch.cuda.Stream() for _ in range(self.num_experts)]

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
        self.gate_network = self.gate_network.to(accelerator.device)
        self.label_smoothing_epsilon = 0.1  
        
        
    def apply_label_smoothing_to_weights(self, gate_weights):

        epsilon = self.label_smoothing_epsilon
        P = self.num_experts
        smoothed_weights = gate_weights * (1 - epsilon) + epsilon / P
        return smoothed_weights
    
    def get_gate_input(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs):
        batch_size = state_tokens.shape[0]
        device = state_tokens.device
        num_experts = len(self.experts)
        hidden_size = 2048
        dtype = torch.bfloat16
        expert_full_hidden_states = [None] * num_experts

        lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs = \
            [x.to(device) for x in [lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs]]
        with torch.no_grad():
            state_tokens_with_mask = torch.cat([state_tokens, action_mask], dim=2)

            lang_cond, img_cond, state_traj = self.experts[0].adapt_conditions(
            lang_tokens, img_tokens, state_tokens_with_mask
            )
        gate_input_buffer = torch.empty((batch_size, self.pred_horizon, num_experts * hidden_size), 
                                        device=device, dtype=dtype)
        
        expert_full_hidden_states = [None] * num_experts
        common_noisy_action = torch.randn(size=(batch_size, self.pred_horizon, self.experts[0].action_dim),
                                      dtype=dtype, device=device)

        for i, expert in enumerate(self.experts):
            with torch.cuda.stream(self.streams[i]):

                expert_hidden = expert.get_last_layer_hidden_states_for_prediction(
                    lang_cond, img_cond, state_traj, action_mask, ctrl_freqs,
                    noisy_action=common_noisy_action
                )
                expert_full_hidden_states[i] = expert_hidden
                gate_input_buffer[:, :, i*hidden_size : (i+1)*hidden_size] = expert_hidden[:, :self.pred_horizon, :]


        for s in self.streams:
            torch.cuda.current_stream().wait_stream(s)
        return gate_input_buffer, expert_full_hidden_states
    
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs):

        gate_input, expert_hidden_states = self.get_gate_input(lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs)
        gate_weights, _ = self.gate_network(gate_input)
        gate_weights = self.apply_label_smoothing_to_weights(gate_weights)
        expert_hidden_states = [h.to(gate_weights.device) for h in expert_hidden_states]
        expert_hidden_states = torch.stack(expert_hidden_states, dim=0)  

        gate_weights_expanded = gate_weights.unsqueeze(-1).unsqueeze(-1)  
        expert_hidden_states_transposed = expert_hidden_states.transpose(0, 1)  
 
        fused_hidden_states = torch.sum(expert_hidden_states_transposed * gate_weights_expanded, dim=1)  
        
        batch_size, horizon, hidden_size = fused_hidden_states.shape
        fused_hidden_states_flat = fused_hidden_states.reshape(-1, hidden_size)  

        fused_action = self.fusion_final_layer(fused_hidden_states_flat)  
        predicted_actions = fused_action.reshape(batch_size, horizon, -1)  
        
        return predicted_actions
    