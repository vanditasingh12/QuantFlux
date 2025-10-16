import numpy as np
import pandas as pd
from collections import deque


class Kelly:
    def __init__(self, market_info: dict[str, str], lookback_period: int = 252, min_trades: int = 30):
        self.lookback_period = lookback_period
        self.min_trades = min_trades
        self.performance_history = deque(maxlen=self.lookback_period)
        self.market_info = market_info
        
        self.kelly_fraction = self._get_kelly_fraction()
        
        print(f"   Kelly fraction: {self.kelly_fraction:.1%}")
        print(f"   Lookback period: {lookback_period} days")
    
    def _get_kelly_fraction(self) -> float:
        region_fractions = {
            "India": 0.25, 
            "US": 0.35, 
            "Europe": 0.30, 
            "UK": 0.30, 
            "Japan": 0.25, 
            "Unknown": 0.20
        }
        return region_fractions.get(self.market_info['region'], 0.25)
    
    def calculate_enhanced_kelly_fractions(self, returns_data: pd.DataFrame, current_allocations: np.ndarray) -> np.ndarray:
        n_assets = len(current_allocations)
        kelly_fractions = np.zeros(n_assets)

        default_allocation = 0.8 / n_assets
        
        for i, asset in enumerate(returns_data.columns):
            asset_returns = returns_data[asset].dropna()
            
            if len(asset_returns) < self.min_trades:
                kelly_fractions[i] = default_allocation
                continue

            short_term_kelly = self._calculate_kelly_for_period(asset_returns.tail(30))
            medium_term_kelly = self._calculate_kelly_for_period(asset_returns.tail(63))
            long_term_kelly = self._calculate_kelly_for_period(asset_returns.tail(126))

            combined_kelly = 0.5 * short_term_kelly + 0.3 * medium_term_kelly + 0.2 * long_term_kelly
            combined_kelly = combined_kelly * self.kelly_fraction
            kelly_fractions[i] = np.clip(combined_kelly, 0.05, 0.4)
        
        total = np.sum(kelly_fractions)
        if total > 0:
            kelly_fractions = kelly_fractions / total
        else:
            kelly_fractions = np.ones(n_assets) / n_assets
        
        return kelly_fractions
    
    def _calculate_kelly_for_period(self, returns: pd.Series) -> float:
        if len(returns) < 10:
            return 0.1
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.1
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = positive_returns.mean()
        avg_loss = abs(negative_returns.mean())
        
        if avg_loss > 0:
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            kelly_f = (b * p - q) / b
            return max(0, kelly_f)
        else:
            return 0.1
    
    def optimize_allocation_with_constraints(self, base_allocation: np.ndarray, kelly_fractions: np.ndarray, 
                                           regime_risk_factor: float, max_position: float = 0.15) -> np.ndarray:
        kelly_weight = 0.3
        combined_allocation = (kelly_weight * kelly_fractions + 
                             (1 - kelly_weight) * base_allocation)
        adjusted_allocation = combined_allocation * regime_risk_factor
        adjusted_allocation = np.clip(adjusted_allocation, 0.05, max_position)
        
        return adjusted_allocation / np.sum(adjusted_allocation)