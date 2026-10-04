import pandas as pd
import numpy as np

def calculate_metrics(strategy_returns: pd.Series, daily_rf: float) -> dict:
    """Calculates performance metrics accounting for event-driven cash dynamics."""
    if strategy_returns.empty or strategy_returns.std() == 0:
        return {"Cumulative Return": "0.00%", "Sharpe Ratio": "0.00", "Max Drawdown": "0.00%"}

    cum_return = (1 + strategy_returns).prod() - 1
    
    # Excess returns over cash hurdle
    excess_returns = strategy_returns - daily_rf
    sharpe_ratio = (excess_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    
    cumulative_equity = (1 + strategy_returns).cumprod()
    rolling_max = cumulative_equity.cummax()
    drawdown = (cumulative_equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        "Cumulative Return": f"{cum_return * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%"
    }

def run_backtest(merged_df: pd.DataFrame, start_date: str, end_date: str, phase_name: str, 
                 holding_days: int = 5, tx_cost_bps: float = 10.0, annual_rf: float = 0.045):
    """
    Backtests an event-driven legal strategy.
    - holding_days: Days to hold position post-ruling/filing.
    - Uninvested days earn the risk-free cash sweep rate.
    """
    mask = (merged_df.index >= start_date) & (merged_df.index <= end_date)
    period_df = merged_df.loc[mask].copy()
    
    if period_df.empty:
        print(f"No data available for {phase_name} ({start_date} to {end_date})")
        return

    daily_rf = annual_rf / 252

    # 1. Generate Raw Event Triggers
    period_df['Signal'] = 0
    period_df.loc[period_df['Alpha_Score'] > 0.5, 'Signal'] = 1    # Long branded innovator
    period_df.loc[period_df['Alpha_Score'] < -0.5, 'Signal'] = -1  # Short branded innovator

    # 2. Apply Event Holding Window (5-day holding period post-catalyst)
    period_df['Target_Position'] = 0
    current_pos = 0
    hold_counter = 0

    target_positions = []
    for signal in period_df['Signal']:
        if signal != 0:
            current_pos = signal
            hold_counter = holding_days
        elif hold_counter > 0:
            hold_counter -= 1
            if hold_counter == 0:
                current_pos = 0
        target_positions.append(current_pos)

    period_df['Target_Position'] = target_positions

    # 3. Execution Lag: Orders execute on T+1
    period_df['Executed_Position'] = period_df['Target_Position'].shift(1).fillna(0)

    # 4. Transaction Cost Penalties on Position Transitions
    pos_changes = period_df['Executed_Position'].diff().abs().fillna(0)
    tx_penalties = pos_changes * (tx_cost_bps / 10000.0)

    # 5. Daily Portfolio Return:
    # Invested portion earns asset return minus costs.
    # Uninvested portion (1 - abs(pos)) earns the cash risk-free rate.
    active_return = period_df['Executed_Position'] * period_df['Daily_Return'] - tx_penalties
    cash_return = (1.0 - period_df['Executed_Position'].abs()) * daily_rf
    period_df['Strategy_Return'] = active_return + cash_return

    metrics = calculate_metrics(period_df['Strategy_Return'].dropna(), daily_rf)
    
    print(f"\n--- {phase_name} Results ({start_date[:4]}-{end_date[:4]}) ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    print("Running event-driven backtest with synchronized catalyst windows...")
    
    np.random.seed(42)
    dates = pd.date_range(start="2016-01-01", end="2026-10-01", freq="B")
    n = len(dates)
    
    # 1. Base stock return with slight positive market drift
    daily_returns = np.random.normal(0.0003, 0.012, n)
    
    # 2. Catalysts occur on ~5% of trading days
    catalyst_days = np.random.binomial(1, 0.05, n)
    raw_signal = np.random.normal(0, 0.6, n)
    alpha_scores = np.where(catalyst_days == 1, raw_signal, 0.0)
    
    # 3. Synchronize the edge:
    # A catalyst on Day T influences returns across Days T+1 through T+3 (post-execution)
    for shift_step in [1, 2, 3]:
        # Roll forward so future days receive the catalyst drift
        daily_returns += np.roll(alpha_scores, shift_step) * 0.003
    
    df = pd.DataFrame({
        'Daily_Return': daily_returns,
        'Alpha_Score': alpha_scores
    }, index=dates)

    run_backtest(df, start_date="2016-01-01", end_date="2023-12-31", phase_name="IN-SAMPLE (TRAIN)")
    run_backtest(df, start_date="2024-01-01", end_date="2026-10-01", phase_name="OUT-OF-SAMPLE (TEST)")