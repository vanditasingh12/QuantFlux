import torch

class Config:
    MAX_DATA_POINTS = 10000
    DDPM_SEQ_LENGTH = 16
    DDPM_BATCH_SIZE = 8
    DDPM_EPOCHS = 10
    DDPM_CHANNELS = [16, 32, 64]
    DDPM_TIMESTEPS = 500
    DDPM_BETA_SCHEDULE = 'cosine'

    RL_WINDOW = 15
    RL_EPISODES = 20
    PPO_EPOCHS = 3
    PPO_CLIP = 0.2
    PPO_LR = 2e-4

    MAX_POSITION_SIZE = 0.25
    MIN_POSITION_SIZE = 0.03
    REBALANCE_THRESHOLD = 0.025
    MIN_DAYS_BETWEEN_REBALANCE = 2
    MAX_DAYS_WITHOUT_REBALANCE = 20
    
    SELECTED_TICKERS = [
        'AAPL', 
        'MSFT', 
        'JPM', 
        'GOOGL', 
        'AMZN', 
        'JNJ', 
        'XOM', 
        'PG', 
        'VZ', 
        'CAT'
    ]
    BENCHMARK_INDEX = '^GSPC'
    
    START_DATE = "2010-01-01"
    END_DATE = "2024-01-01"

    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            print("Using Apple Silicon MPS")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("Using CUDA")
            return torch.device("cuda")
        else:
            print("Using CPU")
            return torch.device("cpu")


def detect_market(benchmark_index, tickers):
    market_info = {"name": "Custom Market", "currency": "$", "region": "Unknown"}

    benchmark_map = {
        '^NSEI': {"name": "NIFTY 50", "currency": "₹", "region": "India"},
        '^SPX': {"name": "S&P 500", "currency": "$", "region": "US"},
        '^GSPC': {"name": "S&P 500", "currency": "$", "region": "US"},
        '^DJI': {"name": "Dow Jones", "currency": "$", "region": "US"},
        '^IXIC': {"name": "NASDAQ", "currency": "$", "region": "US"},
        '^STOXX50E': {"name": "Euro Stoxx 50", "currency": "€", "region": "Europe"},
        '^FTSE': {"name": "FTSE 100", "currency": "£", "region": "UK"},
        '^N225': {"name": "Nikkei 225", "currency": "¥", "region": "Japan"},
        '^HSI': {"name": "Hang Seng", "currency": "HK$", "region": "Hong Kong"},
        '^AXJO': {"name": "ASX 200", "currency": "A$", "region": "Australia"},
    }

    if benchmark_index in benchmark_map:
        market_info.update(benchmark_map[benchmark_index])
    else:
        if any(ticker.endswith('.NS') for ticker in tickers):
            market_info = {"name": "Indian Market", "currency": "₹", "region": "India"}
        elif any(ticker.endswith('.L') for ticker in tickers):
            market_info = {"name": "UK Market", "currency": "£", "region": "UK"}
        elif any(ticker.endswith('.T') for ticker in tickers):
            market_info = {"name": "Japanese Market", "currency": "¥", "region": "Japan"}
        elif any(ticker.endswith('.AS') or ticker.endswith('.PA') or ticker.endswith('.DE') for ticker in tickers):
            market_info = {"name": "European Market", "currency": "€", "region": "Europe"}
    return market_info
