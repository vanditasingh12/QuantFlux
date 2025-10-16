import numpy as np
import pandas as pd

from market.regime_detector import RegimeDetector

class FeatureCalculator:
    def __init__(self, market_info: dict[str, str], min_periods: int = 30):
        self.min_periods = min_periods
        self.market_info = market_info
        self.regime_detector = RegimeDetector(market_info)
    
    def calculate_enhanced_features(self, data: pd.DataFrame, end_idx: int) -> pd.Series:
        if end_idx < self.min_periods:
            return self._get_default_features(data)
        
        historical_data = data.iloc[max(0, end_idx - 100):end_idx + 1]
        features = {}
        all_returns = []
        
        for col in data.columns:
            col_data = historical_data[col]
            col_returns = col_data.pct_change().dropna()
            all_returns.append(col_returns)
            
            #Momentum
            features[f'{col}_momentum_5d'] = self.calculate_momentum(col_data, 5)
            features[f'{col}_momentum_10d'] = self.calculate_momentum(col_data, 10)
            features[f'{col}_momentum_20d'] = self.calculate_momentum(col_data, 20)
            
            #Volatility
            features[f'{col}_vol_5d'] = self.calculate_volatility(col_returns, 5)
            features[f'{col}_vol_20d'] = self.calculate_volatility(col_returns, 20)
            
            #Technical indicators
            features[f'{col}_rsi'] = (self.calculate_rsi(col_data) - 50) / 50
            features[f'{col}_sma_ratio'] = self.calculate_sma_ratio(col_data, 20)
            features[f'{col}_trend_strength'] = self.calculate_trend_strength(col_data, 20)
        
        if all_returns and len(all_returns[0]) > 0:
            market_returns = pd.concat(all_returns, axis=1).mean(axis=1)
            
            features['market_momentum_5d'] = market_returns.tail(5).mean() if len(market_returns) >= 5 else 0
            features['market_momentum_20d'] = market_returns.tail(20).mean() if len(market_returns) >= 20 else 0
            features['market_vol_5d'] = market_returns.tail(5).std() if len(market_returns) >= 5 else 0.02
            features['market_vol_20d'] = market_returns.tail(20).std() if len(market_returns) >= 20 else 0.02
            
            features.update(self._calculate_correlation_features(all_returns))
            
            #Regime detection
            regime_info = self.regime_detector.detect_current_regime(market_returns)
            features.update({
                'bull_prob': regime_info['bull_prob'],
                'bear_prob': regime_info['bear_prob'],
                'sideways_prob': regime_info['sideways_prob'],
                'volatile_prob': regime_info['volatile_prob'],
                'regime_risk_factor': self.regime_detector.get_regime_risk_factor(regime_info)
            })
        else:
            features.update(self._get_default_market_features())
        
        return pd.Series(features).fillna(0.0)
    
    def _calculate_correlation_features(self, all_returns: list[pd.Series]) -> dict[str, float]:
        if len(all_returns) <= 1:
            return {'correlation_avg': 0.0, 'correlation_max': 0.0, 'correlation_dispersion': 0.0}
        
        returns_df = pd.concat(all_returns, axis=1)
        if len(returns_df) < 10:
            return {'correlation_avg': 0.0, 'correlation_max': 0.0, 'correlation_dispersion': 0.0}
        
        corr_matrix = returns_df.tail(30).corr()
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        correlations = corr_matrix.values[mask]
        
        return {
            'correlation_avg': np.mean(correlations),
            'correlation_max': np.max(correlations),
            'correlation_dispersion': np.std(correlations)
        }
    
    def calculate_momentum(self, prices: pd.Series, period: int = 10) -> float:
        if len(prices) < period:
            return 0.0
        return (prices.iloc[-1] / prices.iloc[-period] - 1)
    
    def calculate_volatility(self, returns: pd.Series, period: int = 20) -> float:
        if len(returns) < period:
            return 0.02
        return returns.tail(period).std()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_sma_ratio(self, prices: pd.Series, period: int = 20) -> float:
        if len(prices) < period:
            return 0.0
        sma = prices.tail(period).mean()
        return (prices.iloc[-1] / sma - 1) if sma > 0 else 0
    
    def calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> float:
        if len(prices) < period:
            return 0.0
        returns = prices.pct_change().dropna().tail(period)
        return (returns > 0).mean() - 0.5
    
    def _get_default_features(self, data: pd.DataFrame) -> pd.Series:
        n_assets = len(data.columns)
        feature_names = []
        for col in data.columns:
            feature_names.extend([
                f'{col}_momentum_5d', f'{col}_momentum_10d', f'{col}_momentum_20d',
                f'{col}_vol_5d', f'{col}_vol_20d', f'{col}_rsi', f'{col}_sma_ratio', f'{col}_trend_strength'
            ])
        feature_names.extend([
            'market_momentum_5d', 'market_momentum_20d', 'market_vol_5d', 'market_vol_20d',
            'correlation_avg', 'correlation_max', 'correlation_dispersion',
            'bull_prob', 'bear_prob', 'sideways_prob', 'volatile_prob', 'regime_risk_factor'
        ])
        return pd.Series(0.0, index=feature_names)
    
    def _get_default_market_features(self) -> dict[str, float]:
        return {
            'market_momentum_5d': 0, 'market_momentum_20d': 0,
            'market_vol_5d': 0.02, 'market_vol_20d': 0.02,
            'correlation_avg': 0, 'correlation_max': 0, 'correlation_dispersion': 0,
            'bull_prob': 0.25, 'bear_prob': 0.25, 'sideways_prob': 0.25, 'volatile_prob': 0.25,
            'regime_risk_factor': 1.0
        }