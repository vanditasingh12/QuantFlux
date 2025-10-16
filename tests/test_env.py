from environment.portfolio_env import PortfolioEnv
import pandas as pd
import numpy as np

def test_env_reset_and_step():
    dummy_data = pd.DataFrame(np.random.rand(100, 5), columns=[f"Asset{i}" for i in range(5)])
    market_info = {'region': 'US', 'name': 'S&P 500', 'currency': '$'}
    env = PortfolioEnv(dummy_data, market_info)

    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]

    action = np.array([0.2]*5)
    next_obs, reward, done, _, info = env.step(action)
    assert isinstance(reward, float)
    assert not np.isnan(reward)
