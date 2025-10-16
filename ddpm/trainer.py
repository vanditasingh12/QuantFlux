import torch
import torch.nn.functional as F

class DDPMTrainer:    
    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.loss_history = []
        
    def train_step(self, batch, market_conditions=None):
        self.model.train()
        x = batch.to(self.device)
        batch_size = x.size(0)
        
        try:
            t = self.model.sample_timesteps(batch_size)
            
            noise = torch.randn_like(x) * self.model.volatility_scale
            x_noisy = self.model.add_noise(x, t, noise)
            
            if market_conditions is not None:
                market_conditions = market_conditions.to(self.device)
            
            predicted_noise = self.model(x_noisy, t, market_conditions)
            base_loss = F.mse_loss(predicted_noise, noise, reduction='none')
            
            weights = 1.0 + 0.5 * torch.abs(x).mean(dim=-1, keepdim=True)
            weighted_loss = (base_loss * weights).mean()
            
            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            return weighted_loss.item()
            
        except Exception as e:
            print(f"   DDPM training step failed: {e}")
            if market_conditions is not None:
                print(f"   Market conditions shape: {market_conditions.shape}")
            raise e
    
    def train_epoch(self, dataloader, market_conditions=None):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            batch_market_conditions = None
            if market_conditions is not None:
                batch_market_conditions = market_conditions[num_batches:num_batches+len(batch)]
            
            loss = self.train_step(batch, batch_market_conditions)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.loss_history.append(avg_loss)
        self.scheduler.step()
        return avg_loss