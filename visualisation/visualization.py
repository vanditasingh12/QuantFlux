import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_universal_performance_graph(portfolio_values, benchmark_values, portfolio_dates, benchmark_dates, 
                                       tickers, market_info, benchmark_name):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2.5, 1]})
    plt.style.use('default')  # clean, white background

    portfolio_normalized = np.array(portfolio_values) / portfolio_values[0]
    benchmark_normalized = np.array(benchmark_values) / benchmark_values[0]

    ax1.plot(portfolio_dates, portfolio_normalized, label='Fintrix', color='blue', linestyle='-')
    ax1.plot(benchmark_dates, benchmark_normalized, label=benchmark_name, color='orange', linestyle='--')

    ax1.set_title(f'Cumulative Returns vs {benchmark_name}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'Normalized Price ({market_info["currency"]})')
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper right', frameon=True)

    # Performance annotation box moved to lower left
    final_portfolio_return = (portfolio_normalized[-1] - 1) * 100
    final_benchmark_return = (benchmark_normalized[-1] - 1) * 100
    outperformance = final_portfolio_return - final_benchmark_return

    ax1.text(0.01, 0.05,
         f'Fintrix: {final_portfolio_return:+.1f}%\n'
         f'{benchmark_name}: {final_benchmark_return:+.1f}%\n'
         f'Outperformance: {outperformance:+.1f}%',
         transform=ax1.transAxes,
         verticalalignment='bottom',
         horizontalalignment='left',
         fontsize=9,
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4', alpha=0.9))

    if hasattr(create_universal_performance_graph, 'final_allocation') and create_universal_performance_graph.final_allocation is not None:
        final_alloc = create_universal_performance_graph.final_allocation
        clean_labels = [ticker.replace('.NS', '').replace('.L', '').replace('.T', '')
                        .replace('.AS', '').replace('.PA', '').replace('.DE', '') for ticker in tickers]
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(tickers)))

        ax2.pie(final_alloc, labels=clean_labels, autopct='%1.1f%%', startangle=90,
                colors=colors, textprops={'fontsize': 9})
        ax2.set_title(f'Final Asset Allocation ({market_info["region"]})')

    plt.tight_layout()
    plt.show()
