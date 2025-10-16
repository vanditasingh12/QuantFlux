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

    print("Configuration:")
    print(f"  Market: {market_info['region']} ({market_info['name']})")
    print(f"  Assets: {len(SELECTED_TICKERS)} stocks")
    print(f"  Benchmark: {BENCHMARK_INDEX}")
    print(f"  Currency: {market_info['currency']}")
    print(f"  Date Range: {START_DATE} to {END_DATE}")

    config = Config()
    device = config.get_device()

    print("\nData Collection")
    processor = DataProcessor(market_info)
    stock_data, benchmark_data = processor.download_universal_data(
        SELECTED_TICKERS, BENCHMARK_INDEX, START_DATE, END_DATE
    )
    print(MemoryManager.get_memory_usage())

    print("\nTraining DDPM")
    try:
        ddpm_model = processor.train_ddpm(stock_data, device)
        if processor.ddpm_trainer:
            for epoch, loss in enumerate(processor.ddpm_trainer.loss_history, 1):
                print(f"  DDPM Epoch {epoch}/{config.DDPM_EPOCHS} - Loss: {loss:.4f}")
    except Exception as e:
        ddpm_model = None

    print("\nSynthetic Data Generation")
    try:
        if processor.data_generator:
            training_data = processor.generate_synthetic_training_data(stock_data, num_synthetic_days=80)
        else:
            raise ValueError
    except Exception as e:
        training_data = processor._enhanced_fallback_synthetic_data(stock_data, 80)

    print(f"Training dataset: {len(training_data)} samples")
    MemoryManager.clear_memory()

    print("\nReinforcement Learning")
    env = PortfolioEnv(training_data, market_info, window=config.RL_WINDOW)
    common_indices = training_data.index.intersection(benchmark_data.index)
    if len(common_indices) > 0:
        benchmark_returns = benchmark_data.loc[common_indices]
        if isinstance(benchmark_returns, pd.DataFrame):
            benchmark_returns = benchmark_returns.iloc[:, 0]
        env.set_benchmark_returns(benchmark_returns)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, action_dim, device, market_info)

    for episode in range(config.RL_EPISODES):
        obs, _ = env.reset()
        done = False
        step_count = 0
        while not done:
            historical_returns = training_data.pct_change().iloc[max(0, env.current_step - 126):env.current_step] if env.current_step > 126 else None
            action, log_prob, value = agent.get_action(obs, historical_returns)
            next_obs, reward, done, _, info = env.step(action)
            agent.store_transition(obs, action, reward, value, log_prob, done)
            obs = next_obs
            step_count += 1
        final_value = agent.value_net(torch.FloatTensor(obs).unsqueeze(0).to(device)).item()
        agent.update(final_value)
        print(f"Episode {episode+1}/{config.RL_EPISODES} - Steps: {step_count}")
        MemoryManager.clear_memory()

    print("\nEvaluation on Real Data")
    test_data = stock_data.tail(200)
    test_benchmark = benchmark_data.tail(200)
    test_env = PortfolioEnv(test_data, market_info, window=config.RL_WINDOW)

    test_obs, _ = test_env.reset()
    done = False
    portfolio_values = []
    test_dates = []
    allocations = []

    while not done:
        historical_returns = test_data.pct_change().iloc[max(0, test_env.current_step - 63):test_env.current_step] if test_env.current_step > 63 else None
        action, _, _ = agent.get_action(test_obs, historical_returns, exploration_factor=0.01)
        allocations.append(action)
        test_obs, _, done, _, info = test_env.step(action)
        portfolio_values.append(info['portfolio_value'])
        if test_env.current_step < len(test_data):
            test_dates.append(test_data.index[test_env.current_step])

    benchmark_values = []
    aligned_benchmark = test_benchmark.loc[test_dates[0]:test_dates[-1]]
    if isinstance(aligned_benchmark, pd.DataFrame):
        aligned_benchmark = aligned_benchmark.iloc[:, 0]
    benchmark_values = aligned_benchmark.tolist()
    benchmark_dates = aligned_benchmark.index.tolist()

    create_universal_performance_graph.final_allocation = allocations[-1] if allocations else None
    create_universal_performance_graph(
        portfolio_values=portfolio_values,
        benchmark_values=benchmark_values,
        portfolio_dates=test_dates,
        benchmark_dates=benchmark_dates,
        tickers=SELECTED_TICKERS,
        market_info=market_info,
        benchmark_name=market_info['name']
    )

if __name__ == "__main__":
    print("Configuration:")
    print(f"  Stocks: {len(SELECTED_TICKERS)} assets")
    print(f"  Benchmark: {BENCHMARK_INDEX}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    results = run_enhanced_trading_experiment()

    print("\nFinal Results:")
    print("=" * 60)
    print(f"Portfolio Return: {results['portfolio_return']:.2f}%")
    print(f"Benchmark Return: {results['benchmark_return']:.2f}%")
    print(f"Outperformance: {results['outperformance']:+.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Transaction Costs: {results['total_transaction_costs']:.3f}%")
    print(f"Final Portfolio Value: {results['final_portfolio_value']:.4f}")
    print("=" * 60)
