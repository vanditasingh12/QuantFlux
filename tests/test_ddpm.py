import torch
from ddpm.ddpm import DDPM
from config.config import Config

def test_ddpm_forward():
    config = Config()
    market_info = {'region': 'US', 'name': 'S&P 500', 'currency': '$'}
    model = DDPM(n_assets=5, seq_length=16, market_info=market_info)
    x = torch.randn(2, 5, 16)
    t = torch.randint(0, 200, (2,))
    cond = torch.rand(2, 4)

    out = model(x, t, cond)
    assert out.shape == x.shape
