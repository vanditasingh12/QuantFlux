import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config.config import Config
from market.kelly import Kelly
from market.market_timing import MarketTiming


class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device, market_info: dict[str, str]):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = Config()
        self.market_info = market_info
        
        self.max_position = self.config.MAX_POSITION_SIZE
        self.min_position = self.config.MIN_POSITION_SIZE
        
        hidden_size = max(64, min(256, action_dim * 16))
        
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
        
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        ).to(device)
        
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.PPO_LR)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.config.PPO_LR)
        
        # Enhanced components
        self.kelly_sizer = Kelly(market_info)
        self.market_timer = MarketTiming(market_info)
        
        self.memory = {
            'obs': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': [], 'dones': []
        }
        
        print(f"   Max position size: {self.max_position:.1%}")
        print(f"   Min position size: {self.min_position:.1%}")
        print(f"   Policy params: {sum(p.numel() for p in self.policy_net.parameters()):,}")
    
    def get_action(self, obs, historical_returns=None, exploration_factor=0.1):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.policy_net(obs_tensor)
            value = self.value_net(obs_tensor)
        
        base_allocation = probs.squeeze(0).cpu().numpy()
        
        constrained_allocation = self._apply_diversification_constraints(base_allocation)
        
        if historical_returns is not None and len(historical_returns) > 50:
            try:
                kelly_fractions = self.kelly_sizer.calculate_enhanced_kelly_fractions(
                    historical_returns.tail(126), constrained_allocation
                )
                
                regime_risk_factor = 1.0
                if len(obs) > self.action_dim + 10:
                    regime_risk_factor = obs[-1] if obs[-1] > 0.1 else 1.0
                
                dynamic_max = self.max_position
                if len(obs) > self.action_dim + 15:
                    if obs[-5] > 0.65:  
                        dynamic_max = min(0.35, self.max_position + 0.07)  

                final_allocation = self.kelly_sizer.optimize_allocation_with_constraints(
                constrained_allocation, kelly_fractions, regime_risk_factor, dynamic_max
                )

            except Exception:
                final_allocation = constrained_allocation
        else:
            final_allocation = constrained_allocation
        
        if exploration_factor > 0:
            noise_scale = 0.03 if self.market_info['region'] in ['Japan', 'Europe'] else 0.04
            noise = np.random.normal(0, exploration_factor * noise_scale, size=final_allocation.shape)
            final_allocation = final_allocation + noise
            final_allocation = self._apply_diversification_constraints(final_allocation)
        

        if len(obs) > self.action_dim + 15:  
            try:
                regime_info = {
                    'bull_prob': obs[-5] if len(obs) > self.action_dim + 15 else 0.25,
                    'bear_prob': obs[-4] if len(obs) > self.action_dim + 15 else 0.25,
                    'volatile_prob': obs[-2] if len(obs) > self.action_dim + 15 else 0.25
                }
                market_vol = obs[-6] if len(obs) > self.action_dim + 15 else 0.02
                
                market_exposure = self.market_timer.get_market_exposure(regime_info, market_vol)
                final_allocation, cash_allocation = self.market_timer.apply_market_timing(final_allocation, market_exposure)
                
            except Exception:
                pass 
        
        final_allocation = final_allocation / np.sum(final_allocation)
        log_prob = torch.log(torch.sum(probs * torch.FloatTensor(final_allocation).unsqueeze(0).to(self.device), dim=1) + 1e-8)
        
        return final_allocation, log_prob.item(), value.item()
    
    def _apply_diversification_constraints(self, allocation: np.ndarray) -> np.ndarray:
        constrained = np.clip(allocation, self.min_position, self.max_position)

        total = np.sum(constrained)
        if total > 0:
            constrained = constrained / total
        else:
            constrained = np.ones_like(allocation) / len(allocation)
        
        return constrained
    
    def store_transition(self, obs, action, reward, value, log_prob, done):
        self.memory['obs'].append(obs)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        advantages = []
        gae = 0
        
        rewards = self.memory['rewards']
        values = self.memory['values'] + [next_value]
        dones = self.memory['dones']
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.memory['values'])]
        return advantages, returns
    
    def update(self, next_value):
        if len(self.memory['obs']) < 5:
            return {'policy_loss': 0, 'value_loss': 0}
        
        advantages, returns = self.compute_gae(next_value)
        
        obs_tensor = torch.FloatTensor(np.array(self.memory['obs'])).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.memory['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        if advantages_tensor.std() > 1e-8:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.config.PPO_EPOCHS):
            probs = self.policy_net(obs_tensor)
            values = self.value_net(obs_tensor).squeeze()
            
            new_log_probs = torch.log(torch.sum(probs * actions_tensor, dim=1) + 1e-8)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.config.PPO_CLIP, 1 + self.config.PPO_CLIP) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values, returns_tensor)
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        self.clear_memory()
        
        return {
            'policy_loss': total_policy_loss / self.config.PPO_EPOCHS,
            'value_loss': total_value_loss / self.config.PPO_EPOCHS
        }
    
    def clear_memory(self):
        for key in self.memory:
            self.memory[key] = []