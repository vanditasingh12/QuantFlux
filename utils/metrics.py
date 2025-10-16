import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=252):
    excess_returns = np.array(returns) - risk_free_rate
    std_dev = np.std(excess_returns)

    if std_dev == 0:
        return 0.0

    mean_return = np.mean(excess_returns)
    sharpe_ratio = (mean_return / std_dev) * np.sqrt(annualization_factor)
    return sharpe_ratio


def calculate_max_drawdown(values):
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / peak
    return np.max(drawdowns)


def calculate_calmar_ratio(annual_return, max_drawdown):
    if max_drawdown == 0:
        return 0.0
    return annual_return / max_drawdown


def calculate_annualized_volatility(returns, annualization_factor=252):
    return np.std(returns) * np.sqrt(annualization_factor)
