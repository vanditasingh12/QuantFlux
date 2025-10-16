import numpy as np
import pandas as pd
from collections import deque


class RegimeDetector:
    def __init__(self, market_info: dict, lookback_period: int = 60):
        self.lookback_period = lookback_period
        self.market_info = market_info
        self.regime_history = deque(maxlen=252)

        self.volatility_multiplier = self._get_volatility_multiplier()
        self.return_multiplier = self._get_return_multiplier()

        base_daily_return = 0.0008
        base_volatility = 0.025

        self.bull_thresholds = {
            'min_return': base_daily_return * self.return_multiplier,
            'max_volatility': base_volatility * self.volatility_multiplier,
            'min_trend_strength': 0.55
        }

        self.bear_thresholds = {
            'max_return': -base_daily_return * self.return_multiplier,
            'min_volatility': base_volatility * 0.8 * self.volatility_multiplier,
            'max_trend_strength': 0.45
        }

        self.volatile_threshold = base_volatility * 1.4 * self.volatility_multiplier

    def _get_volatility_multiplier(self) -> float:
        region_multipliers = {
            "India": 1.2,
            "US": 1.0,
            "Europe": 0.9,
            "UK": 0.95,
            "Japan": 0.85,
            "Unknown": 1.0
        }
        return region_multipliers.get(self.market_info['region'], 1.0)

    def _get_return_multiplier(self) -> float:
        region_multipliers = {
            "India": 1.1,
            "US": 1.0,
            "Europe": 0.9,
            "UK": 0.9,
            "Japan": 0.8,
            "Unknown": 1.0
        }
        return region_multipliers.get(self.market_info['region'], 1.0)

    def detect_current_regime(self, returns: pd.Series) -> dict:
        if len(returns) < self.lookback_period:
            return {
                'regime': 'sideways',
                'confidence': 0.5,
                'bull_prob': 0.25,
                'bear_prob': 0.25,
                'sideways_prob': 0.25,
                'volatile_prob': 0.25
            }

        recent_returns = returns.tail(self.lookback_period)
        avg_return = recent_returns.mean()
        volatility = recent_returns.std()
        trend_strength = (recent_returns > 0).mean()

        bull_score = 0
        if avg_return > self.bull_thresholds['min_return']:
            bull_score += 0.4
        if volatility < self.bull_thresholds['max_volatility']:
            bull_score += 0.3
        if trend_strength > self.bull_thresholds['min_trend_strength']:
            bull_score += 0.3

        bear_score = 0
        if avg_return < self.bear_thresholds['max_return']:
            bear_score += 0.4
        if volatility > self.bear_thresholds['min_volatility']:
            bear_score += 0.3
        if trend_strength < self.bear_thresholds['max_trend_strength']:
            bear_score += 0.3

        volatile_score = 0
        if volatility > self.volatile_threshold:
            volatile_score += 0.6
        if abs(avg_return) < 0.0005:
            volatile_score += 0.4

        sideways_score = max(0, 1 - bull_score - bear_score - volatile_score)
        total_score = bull_score + bear_score + volatile_score + sideways_score + 1e-8

        probs = {
            'bull_prob': bull_score / total_score,
            'bear_prob': bear_score / total_score,
            'volatile_prob': volatile_score / total_score,
            'sideways_prob': sideways_score / total_score
        }

        regime = max(probs, key=probs.get).replace('_prob', '')
        confidence = probs[f'{regime}_prob']

        result = {
            'regime': regime,
            'confidence': confidence,
            'avg_return': avg_return,
            'volatility': volatility,
            'trend_strength': trend_strength,
            **probs
        }

        self.regime_history.append(result)
        return result

    def get_regime_risk_factor(self, regime_info: dict) -> float:
        regime = regime_info['regime']
        confidence = regime_info['confidence']

        risk_factors = {
            'bull': 1.25,
            'bear': 0.75,
            'sideways': 0.9,
            'volatile': 0.65
        }

        if self.market_info['region'] in ['India', 'Unknown']:
            risk_factors = {k: v * 0.9 for k, v in risk_factors.items()}

        base_factor = risk_factors.get(regime, 1.0)
        adjusted = 1.0 + (base_factor - 1.0) * confidence
        return np.clip(adjusted, 0.5, 1.5)
