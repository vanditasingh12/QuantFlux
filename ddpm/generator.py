import torch

class DataGenerator:    
    def __init__(self, ddpm_model, device, market_info):
        self.model = ddpm_model
        self.device = device
        self.market_info = market_info
        
    def generate_sequences(self, num_sequences, market_conditions=None, guidance_scale=1.0):
        self.model.eval()
        
        with torch.no_grad():
            shape = (num_sequences, self.model.n_assets, self.model.seq_length)
            x = torch.randn(shape, device=self.device) * self.model.volatility_scale
            
            if market_conditions is not None:
                market_conditions = market_conditions.to(self.device)
                if market_conditions.size(0) != num_sequences:
                    market_conditions = market_conditions.repeat(num_sequences, 1)
            
            for t in reversed(range(self.model.timesteps)):
                t_tensor = torch.full((num_sequences,), t, device=self.device, dtype=torch.long)
                
                predicted_noise = self.model(x, t_tensor, market_conditions)
                
                if guidance_scale != 1.0 and market_conditions is not None:
                    try:
                        uncond_noise = self.model(x, t_tensor, None)
                        predicted_noise = uncond_noise + guidance_scale * (predicted_noise - uncond_noise)
                    except:
                        pass
                
                x = self.model.denoise_step(x, t, predicted_noise, market_conditions)
        
        return x.cpu()
