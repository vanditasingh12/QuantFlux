def clean_ticker_name(ticker: str) -> str:
    return (
        ticker.replace('.NS', '')
              .replace('.L', '')
              .replace('.T', '')
              .replace('.AS', '')
              .replace('.PA', '')
              .replace('.DE', '')
              .replace('.HK', '')
              .replace('.AX', '')
    )


def is_international_ticker(ticker: str) -> bool:
    return any(ticker.endswith(suffix) for suffix in [
        '.NS', '.L', '.T', '.AS', '.PA', '.DE', '.HK', '.AX'
    ])
