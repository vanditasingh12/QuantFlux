import numpy as np
import torch
from models.ppo_agent import PPOAgent

def test_agent_action_shape():
    obs_dim = 20
    action_dim = 5
    obs = np.random.rand(obs_dim).astype(np.float32)
    market_info = {'region': 'US', 'name': 'S&P 500', 'currency': '$'}

    agent = PPOAgent(obs_dim, action_dim, device=torch.device("cpu"), market_info=market_info)
    action, log_prob, value = agent.get_action(obs)

    assert action.shape[0] == action_dim
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
