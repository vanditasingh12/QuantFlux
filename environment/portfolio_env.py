import numpy as np
import pandas as pd
import gymnasium as gym

from config.config import Config
from data.feature_calculator import FeatureCalculator


class PortfolioEnv(gym.Env):
    def __init__(self, price_data: pd.DataFrame, market_info: dict[str, str], window: int = 15):
        super().__init__()
        self.config = Config()
        self.price_data = price_data
        self.tickers = price_data.columns.tolist()
        self.window = window
        self.market_info = market_info
        
        self.rebalance_threshold = self.config.REBALANCE_THRESHOLD
        self.min_days_between_rebalance = self.config.MIN_DAYS_BETWEEN_REBALANCE
        self.max_days_without_rebalance = self.config.MAX_DAYS_WITHOUT_REBALANCE
        self.days_since_rebalance = 0
        
        cost_map = {
            "India": 0.0008,   
            "US": 0.0005,      
            "Europe": 0.0006,  
            "UK": 0.0006,      
            "Japan": 0.0004,   
            "Unknown": 0.0008  
        }
        self.transaction_cost = cost_map.get(market_info['region'], 0.0008)
        
        self.feature_calculator = FeatureCalculator(market_info)
        
        self.returns = price_data.pct_change().dropna()
        self.max_steps = len(self.returns) - 1
        
        n_assets = len(self.tickers)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(n_assets,), dtype=np.float32)
        
        price_features = window * n_assets
        portfolio_weights = n_assets
        enhanced_asset_features = n_assets * 8  
        market_features = 12  
        
        obs_dim = price_features + portfolio_weights + enhanced_asset_features + market_features
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        print(f"   Transaction cost: {self.transaction_cost:.4f}")
        print(f"   Rebalance threshold: {self.rebalance_threshold:.2%}")
        print(f"   Min days between rebalance: {self.min_days_between_rebalance}")
        
        self.current_step = max(self.window, 60)
        self.portfolio_value = 1.0
        self.prev_weights = np.ones(len(self.tickers)) / len(self.tickers)
        self.episode_reward = 0.0
        self.peak_value = 1.0
        self.regime_history = []
        self.allocation_history = []
        self.rebalance_history = []
        
        self.benchmark_returns = None
    
    def set_benchmark_returns(self, benchmark_returns: pd.Series):
        self.benchmark_returns = benchmark_returns.pct_change().dropna()
    
    def reset(self, seed=None, options=None):
        if seed:
            np.random.seed(seed)
        
        min_start = max(self.window, 60)
        max_start = max(min_start + 1, self.max_steps - 30)
        
        if max_start <= min_start:
            self.current_step = min_start
        else:
            random_range = min(50, max_start - min_start)
            if random_range > 0:
                self.current_step = min_start + np.random.randint(0, random_range)
            else:
                self.current_step = min_start
        
        self.portfolio_value = 1.0
        self.prev_weights = np.ones(len(self.tickers)) / len(self.tickers)
        self.episode_reward = 0.0
        self.peak_value = 1.0
        self.regime_history = []
        self.allocation_history = []
        self.rebalance_history = []
        self.days_since_rebalance = 0
        
        return self._get_enhanced_obs(), {}
    
    def _get_enhanced_obs(self):
        start_idx = max(0, self.current_step - self.window + 1)
        end_idx = self.current_step + 1
        price_window = self.price_data.iloc[start_idx:end_idx]
        
        if len(price_window) < self.window:
            padding_needed = self.window - len(price_window)
            first_price = price_window.iloc[[0]] if len(price_window) > 0 else self.price_data.iloc[[0]]
            padding_df = pd.concat([first_price] * padding_needed, ignore_index=True)
            price_window = pd.concat([padding_df, price_window], ignore_index=True)
        
        price_window = price_window.tail(self.window)
        
        if len(price_window) > 0 and not price_window.iloc[0].isna().any():
            norm_prices = (price_window / price_window.iloc[0] - 1).values.flatten()
        else:
            norm_prices = np.zeros(self.window * len(self.tickers))

        weights = self.prev_weights.copy()
        enhanced_features = self.feature_calculator.calculate_enhanced_features(self.price_data, self.current_step)

        if len(enhanced_features) > 0:
            regime_info = {
                'step': self.current_step,
                'bull_prob': enhanced_features.get('bull_prob', 0.25),
                'bear_prob': enhanced_features.get('bear_prob', 0.25),
                'regime_risk_factor': enhanced_features.get('regime_risk_factor', 1.0)
            }
            self.regime_history.append(regime_info)

        obs_parts = [norm_prices, weights, enhanced_features.values]
        obs = np.concatenate(obs_parts).astype(np.float32)

        expected_dim = self.observation_space.shape[0]
        if len(obs) != expected_dim:
            if len(obs) < expected_dim:
                obs = np.pad(obs, (0, expected_dim - len(obs)), 'constant', constant_values=0)
            else:
                obs = obs[:expected_dim]
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs
    
    def _should_rebalance(self, proposed_weights: np.ndarray) -> tuple[bool, str]:
        weight_drift = np.sum(np.abs(proposed_weights - self.prev_weights))

        threshold = self.rebalance_threshold
        if self.regime_history:
            latest = self.regime_history[-1]
            if latest.get('bull_prob', 0.25) > 0.6:
                threshold = 0.02  
            elif latest.get('volatile_prob', 0.25) > 0.5:
                threshold = 0.035  
                
        max_time_reached = self.days_since_rebalance >= self.max_days_without_rebalance
        min_time_passed = self.days_since_rebalance >= self.min_days_between_rebalance
        threshold_breached = weight_drift >= threshold

        if max_time_reached:
            return True, f"Max time reached ({self.days_since_rebalance} days)"
        elif threshold_breached and min_time_passed:
            return True, f"Threshold breached ({weight_drift:.3f} > {threshold:.3f})"
        else:
            return False, f"No rebalance: drift={weight_drift:.3f}, threshold={threshold:.3f}"
    
    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        action = np.clip(action, 0, 1)
        
        total = np.sum(action)
        if total > 1e-8:
            action = action / total
        else:
            action = np.ones_like(action) / len(action)
        
        should_rebalance, rebalance_reason = self._should_rebalance(action)
        
        if should_rebalance:
            actual_weights = action.copy()
            self.days_since_rebalance = 0
            weight_changes = np.sum(np.abs(actual_weights - self.prev_weights))
            transaction_costs = weight_changes * self.transaction_cost
            
            self.rebalance_history.append({
                'step': self.current_step,
                'reason': rebalance_reason,
                'weight_change': weight_changes,
                'transaction_costs': transaction_costs
            })
        else:
            actual_weights = self.prev_weights.copy()
            transaction_costs = 0.0
            self.days_since_rebalance += 1
        
        self.allocation_history.append({
            'step': self.current_step,
            'proposed_allocation': action.copy(),
            'actual_allocation': actual_weights.copy(),
            'rebalanced': should_rebalance,
            'regime_factor': self.regime_history[-1]['regime_risk_factor'] if self.regime_history else 1.0
        })

        if self.current_step < len(self.returns):
            current_returns = self.returns.iloc[self.current_step].values
            gross_return = np.dot(actual_weights, current_returns)
        else:
            gross_return = 0.0
        
        net_return = gross_return - transaction_costs
        
        reward = self._calculate_reward(net_return, actual_weights, gross_return, transaction_costs)
        
        self.prev_weights = actual_weights.copy()
        self.current_step += 1
        self.episode_reward += reward
        
        self.portfolio_value *= (1 + net_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        done = self.current_step >= self.max_steps
        
        info = {
            'portfolio_value': self.portfolio_value,
            'net_return': net_return,
            'gross_return': gross_return,
            'transaction_costs': transaction_costs,
            'weights': actual_weights.copy(),
            'drawdown': drawdown,
            'rebalanced': should_rebalance,
            'days_since_rebalance': self.days_since_rebalance,
            'regime_info': self.regime_history[-1].copy() if self.regime_history else {}
        }
        
        return self._get_enhanced_obs(), reward, done, False, info
    
    def _calculate_reward(self, net_return: float, weights: np.ndarray, 
                                  gross_return: float, transaction_costs: float) -> float:
    
        reward = 0.0
        current_regime = self.regime_history[-1] if self.regime_history else {'bull_prob': 0.25, 'bear_prob': 0.25}
    
        # 1. Risk-adjusted return component with bull boost
        bull_bonus = 1.0
        if current_regime.get('bull_prob', 0.0) > 0.6:
            bull_bonus = 1.1  

        if net_return > 0:
            reward += 0.6 * np.log(1 + net_return) * bull_bonus
        else:
            if net_return > -0.08:
                reward += 0.6 * 0.85 * net_return
            else:
                reward += 0.6 * (-0.5 + 0.85 * net_return)
    
        # 2. Diversification reward 
        herfindahl_index = np.sum(weights ** 2)
        max_diversification = 1 - 1/len(weights)
        diversification_score = (1 - herfindahl_index) / max_diversification
        reward += 0.08 * diversification_score  
    
        # 3. Benchmark tracking component 
        if self.benchmark_returns is not None and self.current_step < len(self.benchmark_returns):
            benchmark_return = self.benchmark_returns.iloc[self.current_step]
            excess_return = net_return - benchmark_return
            reward += 0.35 * excess_return * 30  
    
        # 4. Risk management 
        portfolio_volatility = self._estimate_portfolio_volatility(weights)
        target_volatility = 0.18 if self.market_info['region'] == 'India' else 0.15
    
        if portfolio_volatility < target_volatility:
            reward += 0.07 * (target_volatility - portfolio_volatility) * 10
        else:
            reward -= 0.07 * (portfolio_volatility - target_volatility) * 5
    
        # 5. Transaction cost penalty 
        reward -= 0.0 * transaction_costs * 10
    
        # 6. Regime-aware bonuses 
        if current_regime['bull_prob'] > 0.6 and net_return > 0:
            reward += 0.05 * net_return * 7  
        elif current_regime['bear_prob'] > 0.6 and net_return > -0.02:
            reward += 0.05 * 0.03  
    
        # 7. Momentum bonus 
        if abs(net_return) > 0.005:  
            reward += 0.05 * abs(net_return) * 5
            
        # 8. Drawdown penalty 
        if self.peak_value > 1.0:
            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            if drawdown > 0.05:
                reward -= 0.05 * (drawdown - 0.05) * 10  
                
        # 9. Penalize rapid allocation changes 
        if len(self.allocation_history) > 2:
            prev_alloc = self.allocation_history[-2]['actual_allocation']
            delta = np.sum(np.abs(np.array(weights) - np.array(prev_alloc)))
            reward -= 0.02 * delta  

        return np.clip(reward, -2.0, 3.0) 
    
    def _estimate_portfolio_volatility(self, weights: np.ndarray) -> float:
        if self.current_step < 20:
            return 0.15 
        
        recent_returns = self.returns.iloc[max(0, self.current_step-20):self.current_step]
        if len(recent_returns) > 1:
            portfolio_returns = (recent_returns * weights).sum(axis=1)
            return portfolio_returns.std() * np.sqrt(252)  
        else:
            return 0.15