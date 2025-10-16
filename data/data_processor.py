import numpy as np
import pandas as pd
import yfinance as yf

import torch
from torch.utils.data import DataLoader, TensorDataset

from config.config import Config
from utils.memory import MemoryManager
from market.regime_detector import RegimeDetector

from ddpm.ddpm import DDPM
from ddpm.trainer import DDPMTrainer
from ddpm.generator import DataGenerator
from data.feature_calculator import FeatureCalculator


class DataProcessor:
    def __init__(self, market_info: dict[str, str]):
        self.config = Config()
        self.market_info = market_info
        self.feature_calculator = FeatureCalculator(market_info)
        self.ddpm = None
        self.ddpm_trainer = None
        self.data_generator = None
    
    def download_universal_data(self, tickers: list[str], benchmark: str, 
                               start: str = "2022-01-01", end: str = "2024-01-01"):
        print(f" Selected stocks: {len(tickers)}")
        print(f" Benchmark: {benchmark} ({self.market_info['name']})")
        print(f" Currency: {self.market_info['currency']}")
        print(MemoryManager.get_memory_usage())
        
        try:
            data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
            
            if isinstance(data, pd.Series):
                data = data.to_frame()
                
            benchmark_data = yf.download(benchmark, start=start, end=end, auto_adjust=True)["Close"]
            
            if len(data) > self.config.MAX_DATA_POINTS:
                data = data.tail(self.config.MAX_DATA_POINTS)
                benchmark_data = benchmark_data.tail(self.config.MAX_DATA_POINTS)
            
            data = data.ffill().bfill().dropna()
            benchmark_data = benchmark_data.ffill().bfill()
            
            common_dates = data.index.intersection(benchmark_data.index)
            data = data.loc[common_dates]
            benchmark_data = benchmark_data.loc[common_dates]
            
            for ticker in tickers:
                if ticker in data.columns and len(data) >= 21:
                    recent_return = (data[ticker].iloc[-1] / data[ticker].iloc[-21] - 1) * 100
                    clean_name = ticker.replace('.NS', '').replace('.L', '').replace('.T', '').replace('.AS', '').replace('.PA', '').replace('.DE', '')
                    print(f"   {clean_name}: {recent_return:+.1f}%")
            
            if len(benchmark_data) >= 21:
                if isinstance(benchmark_data, pd.DataFrame):
                    benchmark_col = benchmark_data.columns[0]
                    benchmark_return = (benchmark_data[benchmark_col].iloc[-1] / benchmark_data[benchmark_col].iloc[-21] - 1) * 100
                else:
                    benchmark_return = (benchmark_data.iloc[-1] / benchmark_data.iloc[-21] - 1) * 100
                print(f"   {self.market_info['name']}: {benchmark_return:+.1f}%")
            
            return data, benchmark_data
            
        except Exception as e:
            raise
    
    def prepare_ddpm_training_data(self, data: pd.DataFrame):        
        returns = data.pct_change().dropna()
        
        mean_return = returns.mean()
        std_return = returns.std()
        normalized_returns = (returns - mean_return) / (std_return + 1e-8)
        
        sequences = []
        regime_conditions = []
        
        regime_detector = RegimeDetector(self.market_info)
        
        for i in range(self.config.DDPM_SEQ_LENGTH, len(normalized_returns)):
            seq = normalized_returns.iloc[i-self.config.DDPM_SEQ_LENGTH:i].values.T 
            sequences.append(seq)
            
            period_returns = returns.iloc[:i]
            regime_info = regime_detector.detect_current_regime(period_returns.mean(axis=1))
            regime_vector = [
                regime_info['bull_prob'], regime_info['bear_prob'], 
                regime_info['sideways_prob'], regime_info['volatile_prob']
            ]
            regime_conditions.append(regime_vector)
        
        sequences = np.array(sequences)
        regime_conditions = np.array(regime_conditions)
        
        print(f" Sequence shape: {sequences.shape}")
        print(f" Regime conditions shape: {regime_conditions.shape}")
        
        self.normalization_params = {'mean': mean_return, 'std': std_return}
        
        return sequences, regime_conditions
    
    def train_ddpm(self, data: pd.DataFrame, device):
        sequences, regime_conditions = self.prepare_ddpm_training_data(data)
        
        self.ddpm = DDPM(
            n_assets=len(data.columns),
            seq_length=self.config.DDPM_SEQ_LENGTH,
            market_info=self.market_info,
            device=device
        ).to(device)
        
        self.ddpm_trainer = DDPMTrainer(self.ddpm, device)
        
        tensor_sequences = torch.FloatTensor(sequences)
        tensor_conditions = torch.FloatTensor(regime_conditions)
        dataset = TensorDataset(tensor_sequences, tensor_conditions)
        dataloader = DataLoader(dataset, batch_size=self.config.DDPM_BATCH_SIZE, shuffle=True)
        
        print(f" Training DDPM ({self.config.DDPM_EPOCHS} epochs)")
        for epoch in range(self.config.DDPM_EPOCHS):
            epoch_loss = 0
            num_batches = 0
            
            for batch_sequences, batch_conditions in dataloader:
                loss = self.ddpm_trainer.train_step(batch_sequences, batch_conditions)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            self.ddpm_trainer.loss_history.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f" Epoch {epoch+1}/{self.config.DDPM_EPOCHS}, Loss: {avg_loss:.6f}")
                print(f" {MemoryManager.get_memory_usage()}")
            
            MemoryManager.clear_memory()
            
        self.data_generator = DataGenerator(self.ddpm, device, self.market_info)
        
        if len(self.ddpm_trainer.loss_history) > 0:
            print(f"Final loss: {self.ddpm_trainer.loss_history[-1]:.6f}")
        return self.ddpm
    
    def generate_synthetic_training_data(self, real_data: pd.DataFrame, num_synthetic_days: int = 60):
        if self.data_generator is None:
            return self._enhanced_fallback_synthetic_data(real_data, num_synthetic_days)
        
        try:
            returns = real_data.pct_change().dropna()
            regime_detector = RegimeDetector(self.market_info)
            recent_regime = regime_detector.detect_current_regime(returns.mean(axis=1))
            
            num_sequences = max(3, num_synthetic_days // self.config.DDPM_SEQ_LENGTH)
            
            market_conditions = torch.tensor([
                recent_regime.get('bull_prob', 0.25), recent_regime.get('bear_prob', 0.25),
                recent_regime.get('sideways_prob', 0.25), recent_regime.get('volatile_prob', 0.25)
            ], dtype=torch.float32).unsqueeze(0).repeat(num_sequences, 1)
            
            synthetic_sequences = self.data_generator.generate_sequences(
                num_sequences, market_conditions=market_conditions, guidance_scale=1.1
            )
            
            synthetic_data_frames = []
            last_date = real_data.index[-1]
            last_prices = real_data.iloc[-1].values
            
            for i in range(num_sequences):
                sequence_returns = synthetic_sequences[i].numpy().T  
                max_daily_move = 0.08 if self.market_info['region'] in ['India', 'Unknown'] else 0.06
                sequence_returns = np.clip(sequence_returns, -max_daily_move, max_daily_move)
                
                price_sequence = []
                current_prices = last_prices.copy()
                
                for t in range(len(sequence_returns)):
                    returns_t = sequence_returns[t]
                    current_prices = current_prices * (1 + returns_t)
                    price_sequence.append(current_prices.copy())
                
                start_date = last_date + pd.Timedelta(days=1 + i * self.config.DDPM_SEQ_LENGTH)
                dates = pd.date_range(start_date, periods=len(price_sequence), freq='D')
                
                synthetic_df = pd.DataFrame(price_sequence, index=dates, columns=real_data.columns)
                synthetic_data_frames.append(synthetic_df)
            
            if synthetic_data_frames:
                all_synthetic = pd.concat(synthetic_data_frames, axis=0)
                all_synthetic = all_synthetic.head(num_synthetic_days)
                
                training_data = pd.concat([real_data, all_synthetic], axis=0)
                
                print(f" Generated {len(all_synthetic)} days of DDPM synthetic data")
                print(f" Training dataset: {len(real_data)} real + {len(all_synthetic)} synthetic = {len(training_data)} total")
                
                return training_data
            else:
                return self._enhanced_fallback_synthetic_data(real_data, num_synthetic_days)
                
        except Exception as e:
            return self._enhanced_fallback_synthetic_data(real_data, num_synthetic_days)
    
    def _enhanced_fallback_synthetic_data(self, real_data: pd.DataFrame, num_synthetic_days: int = 60):
        
        training_data = real_data.copy()
        
        if len(real_data) > 60:
            returns = real_data.pct_change().dropna()
            
            short_returns = returns.tail(10).mean()
            medium_returns = returns.tail(30).mean() 
            long_returns = returns.tail(60).mean()
            
            short_vol = returns.tail(10).std()
            medium_vol = returns.tail(30).std()

            corr_matrix = returns.tail(60).corr()
            
            for i in range(num_synthetic_days):
                regime_detector = RegimeDetector(self.market_info)
                recent_regime = regime_detector.detect_current_regime(returns.mean(axis=1))

                if recent_regime['bull_prob'] > 0.5:
                    base_returns = long_returns * 1.1
                elif recent_regime['bear_prob'] > 0.5:
                    base_returns = long_returns * 0.9
                else:
                    base_returns = medium_returns

                regime_vol_factor = 1.0
                if recent_regime['volatile_prob'] > 0.4:
                    regime_vol_factor = 1.3
                elif recent_regime['bull_prob'] > 0.6:
                    regime_vol_factor = 0.8

                noise = np.random.multivariate_normal(
                    mean=np.zeros(len(real_data.columns)),
                    cov=corr_matrix.values * (medium_vol.values[:, None] * medium_vol.values[None, :]) * regime_vol_factor,
                    size=1
                )[0]
                
                synthetic_returns = base_returns + noise * 0.5

                max_move = 0.08 if self.market_info['region'] in ['India','Unknown'] else 0.06
                synthetic_returns = np.clip(synthetic_returns, -max_move, max_move)

                last_prices = training_data.iloc[-1]
                new_prices = last_prices * (1 + synthetic_returns)

                training_data.loc[len(training_data)] = new_prices
        return training_data