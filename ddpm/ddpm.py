import torch
import torch.nn as nn
import math

from ddpm.embeddings import SinusoidalPositionalEmbedding

class DDPM(nn.Module):    
    def __init__(self, n_assets, seq_length, market_info, device='cpu'):
        super().__init__()
        self.n_assets = n_assets
        self.seq_length = seq_length
        self.timesteps = 200
        self.device = device
        self.market_info = market_info
        
        hidden_dim = 128
        
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(32),
            nn.Linear(32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.market_embed = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(n_assets * seq_length + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_assets * seq_length)
        )
        
        self.register_buffer('betas', self._get_beta_schedule())
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))
        self.volatility_scale = self._get_market_volatility_scale()
    
    def _get_beta_schedule(self):
        return torch.linspace(0.0001, 0.02, self.timesteps)
    
    def _get_market_volatility_scale(self):
        scale_map = {
            "India": 1.2, 
            "US": 1.0, 
            "Europe": 0.9, 
            "UK": 0.9, 
            "Japan": 0.8, 
            "Unknown": 1.1
        }
        return scale_map.get(self.market_info['region'], 1.0)
    
    def forward(self, x, t, market_condition=None):
        batch_size = x.size(0)
        
        x_flat = x.view(batch_size, -1)
        time_emb = self.time_embed(t)
        
        if market_condition is not None:
            market_emb = self.market_embed(market_condition)
            time_emb = time_emb + market_emb

        combined = torch.cat([x_flat, time_emb], dim=1)
        
        output_flat = self.net(combined)
        output = output_flat.view(batch_size, self.n_assets, self.seq_length)
        
        return output
    
    def add_noise(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x) * self.volatility_scale
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)
    
    def denoise_step(self, x_t, t, model_output, market_condition=None):
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_cumprod_prev = self.alphas_cumprod[t-1] if t > 0 else torch.ones_like(alpha_cumprod_t)
        alpha_cumprod_prev = alpha_cumprod_prev.view(-1, 1, 1)
        
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * model_output) / torch.sqrt(alpha_cumprod_t)
        
        posterior_mean = (
            torch.sqrt(alpha_cumprod_prev) * self.betas[t].view(-1, 1, 1) * pred_x0 +
            torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) * x_t
        ) / (1 - alpha_cumprod_t)
        
        if t == 0:
            return posterior_mean
        else:
            posterior_variance = self.betas[t] * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
            noise = torch.randn_like(x_t) * self.volatility_scale
            return posterior_mean + torch.sqrt(posterior_variance.view(-1, 1, 1)) * noise
