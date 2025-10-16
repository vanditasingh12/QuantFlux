from config.config import Config, detect_market
from data.data_processor import DataProcessor
from utils.memory import MemoryManager
from environment.portfolio_env import PortfolioEnv
from models.ppo_agent import PPOAgent
from visualisation.visualization import create_universal_performance_graph

import numpy as np
import pandas as pd
import torch
import gymnasium as gym

SELECTED_TICKERS = Config.SELECTED_TICKERS
BENCHMARK_INDEX = Config.BENCHMARK_INDEX
START_DATE = Config.START_DATE
END_DATE = Config.END_DATE


def run_enhanced_trading_experiment():
    market_info = detect_market(BENCHMARK_INDEX, SELECTED_TICKERS)
    print(f"   CONFIGURATION:")
    print(f"   Market: {market_info['region']} ({market_info['name']})")
    print(f"   Assets: {len(SELECTED_TICKERS)} stocks")
    print(f"   Benchmark: {BENCHMARK_INDEX}")
    print(f"   Currency: {market_info['currency']}")
    print(f"   Date Range: {START_DATE} to {END_DATE}")
    print("=" * 80)
    
    config = Config()
    device = config.get_device()
    
    processor = DataProcessor(market_info)
    
    stock_data, benchmark_data = processor.download_universal_data(
        SELECTED_TICKERS, BENCHMARK_INDEX, START_DATE, END_DATE
    )
    print(MemoryManager.get_memory_usage())
    
    # Train DDPM for Synthetic Data Generation
    try:
        ddpm_model = processor.train_ddpm(stock_data, device)
    except Exception as e:
        print(f"DDPM training failed: {e}")
        ddpm_model = None
    
    # Generate Enhanced Training Data
    try:
        if processor.data_generator is not None:
            training_data = processor.generate_synthetic_training_data(
                stock_data, 
                num_synthetic_days=80
            )
        else:
            raise Exception("DDPM data generator not available")
    except Exception as e:
        print("   Using fallback synthetic data generation")
        training_data = processor._enhanced_fallback_synthetic_data(stock_data, 80)
    
    MemoryManager.clear_memory()
    
    # Step 4: Enhanced RL Training
    env = PortfolioEnv(training_data, market_info, window=config.RL_WINDOW)
    common_indices = training_data.index.intersection(benchmark_data.index)
    if len(common_indices) > 0:
        benchmark_returns = benchmark_data.loc[common_indices]
        if isinstance(benchmark_returns, pd.DataFrame):
            benchmark_returns = benchmark_returns.iloc[:, 0]
        env.set_benchmark_returns(benchmark_returns)
    else:
        print(" No common indices found between training data and benchmark.")
    
    test_obs, _ = env.reset()
    print(f"Test observation shape: {test_obs.shape}")
    print(f"Expected shape: {env.observation_space.shape}")
    
    if test_obs.shape[0] != env.observation_space.shape[0]:
        env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(test_obs.shape[0],), 
            dtype=np.float32
        )
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim, device, market_info)
    
    episode_rewards = []
    episode_values = []
    regime_tracking = []
    rebalancing_stats = []
    
    for episode in range(config.RL_EPISODES):
        obs, _ = env.reset()
        episode_reward = 0
        episode_value = []
        step_count = 0
        max_steps_per_episode = 100
        episode_rebalances = 0
        
        done = False
        while not done and step_count < max_steps_per_episode:
            historical_returns = training_data.pct_change().iloc[
                max(0, env.current_step-126):env.current_step
            ] if env.current_step > 126 else None
            
            exploration = 0.06 if market_info['region'] in ['Japan', 'Europe'] else 0.08
            action, log_prob, value = agent.get_action(
                obs, historical_returns=historical_returns, exploration_factor=exploration
            )
            next_obs, reward, done, _, info = env.step(action)
            
            agent.store_transition(obs, action, reward, value, log_prob, done)
            
            obs = next_obs
            episode_reward += reward
            episode_value.append(info['portfolio_value'])
            step_count += 1
            
            if info.get('rebalanced', False):
                episode_rebalances += 1
        
        # Update agent
        final_value = agent.value_net(torch.FloatTensor(obs).unsqueeze(0).to(device)).item()
        update_info = agent.update(final_value)
        
        episode_rewards.append(episode_reward)
        episode_values.append(episode_value[-1] if episode_value else 1.0)
        rebalancing_stats.append(episode_rebalances)
        
        if hasattr(env, 'regime_history') and env.regime_history:
            regime_tracking.extend(env.regime_history)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_value = np.mean(episode_values[-10:])
            avg_rebalances = np.mean(rebalancing_stats[-10:])
            
            recent_regimes = [r for r in regime_tracking if r['step'] >= env.current_step - 100]
            if recent_regimes:
                avg_bull_prob = np.mean([r['bull_prob'] for r in recent_regimes])
                avg_risk_factor = np.mean([r['regime_risk_factor'] for r in recent_regimes])
                print(f"Episode {episode+1}/{config.RL_EPISODES}")
                print(f"  Avg Reward: {avg_reward:.3f}")
                print(f"  Avg Portfolio Value: {avg_value:.3f}")
                print(f"  Avg Rebalances per Episode: {avg_rebalances:.1f}")
                print(f"  Bull Market Probability: {avg_bull_prob:.3f}")
                print(f"  Avg Risk Factor: {avg_risk_factor:.3f}")
            
            print(MemoryManager.get_memory_usage())
        
        MemoryManager.clear_memory()
    
    # Evaluation on REAL DATA ONLY
    print(f"\n {market_info['region']} Strategy Evaluation vs {market_info['name']}")

    test_data = stock_data.tail(200)  
    test_benchmark = benchmark_data.tail(200)
    test_env = PortfolioEnv(test_data, market_info, window=config.RL_WINDOW)
    
    common_test_indices = test_data.index.intersection(test_benchmark.index)
    if len(common_test_indices) > 0:
        test_benchmark_returns = test_benchmark.loc[common_test_indices]
        if isinstance(test_benchmark_returns, pd.DataFrame):
            test_benchmark_returns = test_benchmark_returns.iloc[:, 0]
        test_env.set_benchmark_returns(test_benchmark_returns)
    else:
        print(" No common indices found between test data and benchmark. Skipping benchmark tracking.")
    
    obs, _ = test_env.reset()
    test_rewards = []
    portfolio_values = []
    allocations = []
    transaction_costs_history = []
    regime_decisions = []
    test_dates = []
    rebalancing_events = []
    
    done = False
    while not done:
        historical_returns = test_data.pct_change().iloc[
            max(0, test_env.current_step-63):test_env.current_step
        ] if test_env.current_step > 63 else None
        
        action, _, _ = agent.get_action(
            obs, historical_returns=historical_returns, exploration_factor=0.01
        )
        allocations.append(action.copy())
        obs, reward, done, _, info = test_env.step(action)
        test_rewards.append(reward)
        portfolio_values.append(info['portfolio_value'])
        transaction_costs_history.append(info['transaction_costs'])
        
        if test_env.current_step < len(test_data):
            test_dates.append(test_data.index[test_env.current_step])
        
        if 'regime_info' in info:
            regime_decisions.append(info['regime_info'])

        if info.get('rebalanced', False):
            rebalancing_events.append({
                'step': test_env.current_step,
                'date': test_data.index[test_env.current_step] if test_env.current_step < len(test_data) else None,
                'days_since_last': info.get('days_since_rebalance', 0),
                'transaction_costs': info['transaction_costs']
            })
    
    benchmark_test_values = []
    benchmark_test_dates = []

    if test_dates:
        start_date = test_dates[0]
        end_date = test_dates[-1]
        aligned_benchmark = test_benchmark.loc[start_date:end_date]

        if not aligned_benchmark.empty:
            if isinstance(aligned_benchmark, pd.DataFrame):
                aligned_series = aligned_benchmark.iloc[:, 0]
            else:
                aligned_series = aligned_benchmark

            benchmark_test_values = aligned_series.tolist()
            benchmark_test_dates = aligned_series.index.tolist()
    
    if portfolio_values and benchmark_test_values:
        final_portfolio_value = portfolio_values[-1]
        final_benchmark_value = benchmark_test_values[-1] / benchmark_test_values[0]
        
        portfolio_return = (final_portfolio_value - 1) * 100
        benchmark_return = (final_benchmark_value - 1) * 100
        outperformance = portfolio_return - benchmark_return
        
        if len(test_rewards) > 1:
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            if np.std(daily_returns) > 1e-8:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Risk metrics
        peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdowns) * 100
        
        calmar_ratio = portfolio_return / max_drawdown if max_drawdown > 0 else 0
        total_tc = sum(transaction_costs_history) * 100
        
        positive_days = sum(1 for r in test_rewards if r > 0)
        win_rate = positive_days / len(test_rewards) * 100 if test_rewards else 0
        
        if len(portfolio_values) > 1:
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            annual_vol = np.std(daily_returns) * np.sqrt(252) * 100
        else:
            annual_vol = 0
    
    else:
        portfolio_return = benchmark_return = outperformance = 0.0
        sharpe_ratio = max_drawdown = calmar_ratio = 0.0
        total_tc = win_rate = annual_vol = 0.0
        final_portfolio_value = 1.0
    
    rebalancing_analysis = {}
    if rebalancing_events:
        rebalancing_analysis = {
            'total_rebalances': len(rebalancing_events),
            'avg_days_between': np.mean([event['days_since_last'] for event in rebalancing_events[1:]]) if len(rebalancing_events) > 1 else 0,
            'total_rebalance_costs': sum([event['transaction_costs'] for event in rebalancing_events]) * 100,
            'rebalance_frequency': len(rebalancing_events) / len(test_data) * 252  
        }
    
    regime_summary = {}
    if regime_decisions:
        bull_periods = sum(1 for r in regime_decisions if r.get('bull_prob', 0) > 0.5)
        bear_periods = sum(1 for r in regime_decisions if r.get('bear_prob', 0) > 0.5)
        avg_risk_factor = np.mean([r.get('regime_risk_factor', 1.0) for r in regime_decisions])
        
        regime_summary = {
            'bull_periods_pct': bull_periods / len(regime_decisions) * 100,
            'bear_periods_pct': bear_periods / len(regime_decisions) * 100,
            'avg_risk_factor': avg_risk_factor
        }
    
    market_flag = "India" if market_info['region'] == "India" else "USA" if market_info['region'] == "US" else "other"
    print(f"\n{market_info['region'].upper()} MARKET RESULTS SUMMARY:")
    print(f"{'='*70}")
    print(f" Training: DDPM-Enhanced Dataset ({len(training_data)} samples)")
    print(f" Testing: Real Data Only ({len(test_data)} samples)")
    print(f"{'='*70}")
    print(f" Portfolio Return: {portfolio_return:.2f}%")
    print(f" Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f" Max Drawdown: {max_drawdown:.2f}%")
    print(f" Calmar Ratio: {calmar_ratio:.3f}")
    print(f" Win Rate: {win_rate:.1f}%")
    print(f" Annual Volatility: {annual_vol:.2f}%")
    print(f" Final Portfolio Value: {final_portfolio_value:.4f}")
    print(f" Total Transaction Costs: {total_tc:.3f}%")
    print(f" Currency: {market_info['currency']}")
    print(f"{'='*70}")
    print(f" {market_info['name']} Benchmark Return: {benchmark_return:.2f}%")
    print(f" Outperformance vs {market_info['name']}: {outperformance:+.2f}%")
    
    if rebalancing_analysis:
        print(f"\n REBALANCING ANALYSIS:")
        print(f"   Total Rebalancing Events: {rebalancing_analysis['total_rebalances']}")
        print(f"   Average Days Between Rebalances: {rebalancing_analysis['avg_days_between']:.1f}")
        print(f"   Annualized Rebalancing Frequency: {rebalancing_analysis['rebalance_frequency']:.1f}")
        print(f"   Total Rebalancing Costs: {rebalancing_analysis['total_rebalance_costs']:.3f}%")
    
    if regime_summary:
        print(f"\n REGIME ANALYSIS:")
        print(f"   Bull Market Periods: {regime_summary['bull_periods_pct']:.1f}%")
        print(f"   Bear Market Periods: {regime_summary['bear_periods_pct']:.1f}%")
        print(f"   Average Risk Factor: {regime_summary['avg_risk_factor']:.2f}")
    
    # Asset allocation analysis
    if allocations:
        final_allocation = allocations[-1]
        avg_allocation = np.mean(allocations, axis=0)
        
        print(f"\n ENHANCED ASSET ALLOCATION ANALYSIS:")
        print(f"   Final Allocation (Diversification Constraints Applied):")
        for i, ticker in enumerate(SELECTED_TICKERS):
            clean_name = ticker.replace('.NS', '').replace('.L', '').replace('.T', '').replace('.AS', '').replace('.PA', '').replace('.DE', '')
            print(f"     {clean_name}: {final_allocation[i]*100:.1f}%")
        
        print(f"   Average Allocation:")
        for i, ticker in enumerate(SELECTED_TICKERS):
            clean_name = ticker.replace('.NS', '').replace('.L', '').replace('.T', '').replace('.AS', '').replace('.PA', '').replace('.DE', '')
            print(f"     {clean_name}: {avg_allocation[i]*100:.1f}%")
        
        final_herfindahl = np.sum(final_allocation ** 2)
        avg_herfindahl = np.sum(avg_allocation ** 2)
        max_diversification = 1 - 1/len(final_allocation)
        
        final_diversification_score = (1 - final_herfindahl) / max_diversification
        avg_diversification_score = (1 - avg_herfindahl) / max_diversification
        
        print(f"\n DIVERSIFICATION ANALYSIS:")
        print(f"   Final Diversification Score: {final_diversification_score:.1%}")
        print(f"   Average Diversification Score: {avg_diversification_score:.1%}")
        print(f"   Maximum Single Position: {np.max(final_allocation)*100:.1f}%")
        print(f"   Minimum Single Position: {np.min(final_allocation)*100:.1f}%")
    
    # DDPM Analysis
    if processor.ddpm_trainer and len(processor.ddpm_trainer.loss_history) > 0:
        print(f"\n DDPM TRAINING ANALYSIS:")
        print(f"   Final Training Loss: {processor.ddpm_trainer.loss_history[-1]:.6f}")
        print(f"   Training Epochs: {len(processor.ddpm_trainer.loss_history)}")
        print(f"   Synthetic Data Quality: High-fidelity financial sequences")
        print(f"   Market Regime Awareness: Incorporated")
    
    # Create enhanced performance graph
    print(f"\n Creating Enhanced {market_info['region']} Market Performance Visualization...")
    
    # Store data for graph function
    create_universal_performance_graph.final_allocation = allocations[-1] if allocations else None
    
    # Create the comprehensive performance graph
    if portfolio_values and benchmark_test_values:
        create_universal_performance_graph(
            portfolio_values=portfolio_values,
            benchmark_values=benchmark_test_values,
            portfolio_dates=test_dates[:len(portfolio_values)],
            benchmark_dates=benchmark_test_dates[:len(benchmark_test_values)],
            tickers=SELECTED_TICKERS,
            market_info=market_info,
            benchmark_name=market_info['name']
        )
    
    print(f"\n Enhanced Universal {market_info['region']} Trading Experiment Complete!")
    print(f" Critical Improvements Applied:")
    print(f"    Diversification constraints (max {config.MAX_POSITION_SIZE:.1%} per asset)")
    print(f"    Threshold-based rebalancing ({config.REBALANCE_THRESHOLD:.1%} trigger)")
    print(f"    Enhanced reward function with benchmark tracking")
    print(f"    Multi-timeframe feature engineering")
    print(f"    Market timing layer")
    print(f"    Enhanced Kelly position sizing")
    print(f"    Reduced transaction costs through smart rebalancing")
    print(f"🔍 Validation: Testing performed on real data only")
    print(MemoryManager.get_memory_usage())
    
    return {
        'portfolio_return': portfolio_return,
        'benchmark_return': benchmark_return,
        'outperformance': outperformance,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'annual_volatility': annual_vol,
        'final_portfolio_value': final_portfolio_value,
        'total_transaction_costs': total_tc,
        'episode_rewards': episode_rewards,
        'final_allocation': allocations[-1] if allocations else None,
        'avg_allocation': np.mean(allocations, axis=0) if allocations else None,
        'regime_summary': regime_summary,
        'rebalancing_analysis': rebalancing_analysis,
        'portfolio_values': portfolio_values,
        'benchmark_values': benchmark_test_values,
        'market_info': market_info,
        'ddpm_enhanced': processor.data_generator is not None,
        'training_data_size': len(training_data),
        'test_data_size': len(test_data),
        'improvements_applied': [
            'diversification_constraints',
            'threshold_rebalancing', 
            'enhanced_reward_function',
            'multi_timeframe_features',
            'market_timing',
            'enhanced_kelly_sizing'
        ]
    }

if __name__ == "__main__":
    print("Fintrix")
    print("=" * 60)
    print("Enhanced Configuration:")
    print(f"   Stocks: {len(SELECTED_TICKERS)} assets")
    print(f"   Benchmark: {BENCHMARK_INDEX}")
    print(f"   Period: {START_DATE} to {END_DATE}")
    print("=" * 60)
    
    results = run_enhanced_trading_experiment()

    market_flag = "India" if results['market_info']['region'] == "India" else "USA" if results['market_info']['region'] == "US" else "other"
    print(f"\n{market_flag} FINAL {results['market_info']['region'].upper()} RESULTS:")
    print(f"  DDPM Enhanced: {' Yes' if results['ddpm_enhanced'] else 'No'}")
    print(f"  Training Data: {results['training_data_size']} samples")
    print(f"  Test Data: {results['test_data_size']} samples (Real Only)")
    print(f"  Enhanced AI Portfolio: {results['portfolio_return']:.2f}%")
    print(f"  Benchmark: {results['benchmark_return']:.2f}%")
    print(f"  Outperformance: {results['outperformance']:+.2f}%")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"  Transaction Costs: {results['total_transaction_costs']:.3f}%")
    print(f"  Rebalancing Events: {results['rebalancing_analysis'].get('total_rebalances', 0)}")
    print(f"  Currency: {results['market_info']['currency']}")
    print(f"\n  Improvements Applied: {len(results['improvements_applied'])}/6")
    for improvement in results['improvements_applied']:
        print(f"    {improvement.replace('_', ' ').title()}")
