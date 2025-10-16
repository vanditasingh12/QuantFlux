import numpy as np

class MarketTiming:
    def __init__(self, market_info: dict[str, str]):
        self.market_info = market_info
        self.volatility_threshold = 0.25
    
    def get_market_exposure(self, current_regime: dict, market_volatility: float) -> float:
        base_exposure = 1.0

        if current_regime['bull_prob'] > 0.7:
            base_exposure *= 1.4 
        elif current_regime['bull_prob'] > 0.6:
            base_exposure *= 1.25
        elif current_regime['bear_prob'] > 0.8:
            base_exposure *= 0.85
        elif current_regime['volatile_prob'] > 0.6:
            base_exposure *= 0.9

        if market_volatility > self.volatility_threshold:
            vol_reduction = min(0.25, (market_volatility - self.volatility_threshold) * 1.5)
            base_exposure *= (1 - vol_reduction)

        return np.clip(base_exposure, 0.8, 1.5)

    def apply_market_timing(self, base_allocation: np.ndarray, market_exposure: float) -> tuple[np.ndarray, float]:
        scaled_allocation = base_allocation * market_exposure
        cash_allocation = 1.0 - np.sum(scaled_allocation)
        
        return scaled_allocation, max(0, cash_allocation)
