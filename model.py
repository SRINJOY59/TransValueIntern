import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_and_train_lstm(X_train, y_train, epochs=50, batch_size=1):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)
    return model

def forecast_nav(model, last_sequence, num_months):
    forecast = []
    for _ in range(num_months):
        pred = model.predict(last_sequence[np.newaxis, :, :])
        forecast.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred, axis=0)
    return np.array(forecast).reshape(-1, 1)

def calculate_features(df, benchmark_df):
    df['Monthly_Return'] = df['NAV'].pct_change()
    df['4M_Rolling_Return'] = df['NAV'].pct_change(4)
    variance_monthly_returns = df['Monthly_Return'].var()

    df['Cumulative_NAV'] = (1 + df['Monthly_Return']).cumprod()
    df['Rolling_Max'] = df['Cumulative_NAV'].cummax()
    df['Drawdown'] = df['Cumulative_NAV'] - df['Rolling_Max']
    df['Drawdown_Duration'] = df['Drawdown'].ne(0).astype(int).groupby(df['Drawdown'].ne(0).cumsum()).cumsum()
    max_drawdown = df['Drawdown'].min()
    drawdown_duration = df['Drawdown_Duration'].max()

    risk_free_rate = 0.03 / 12
    annualized_return = df['Monthly_Return'].mean() * 12
    annualized_volatility = df['Monthly_Return'].std() * np.sqrt(12)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    downside_returns = df['Monthly_Return'][df['Monthly_Return'] < 0]
    downside_deviation = downside_returns.std() * np.sqrt(12)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation

    # df = df.join(benchmark_df[['Benchmark_Return']])
    cov_matrix = df[['Monthly_Return', 'Benchmark_Return']].dropna().cov()
    beta = cov_matrix.loc['Monthly_Return', 'Benchmark_Return'] / df['Benchmark_Return'].var()

    expense_ratio = 0.01  # Placeholder value

    return {
        'variance_monthly_returns': variance_monthly_returns,
        'max_drawdown': max_drawdown,
        'drawdown_duration': drawdown_duration,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'beta': beta,
        'expense_ratio': expense_ratio
    }


# Generate synthetic benchmark data
np.random.seed(0)
# benchmark_navs = df['NAV'] * (1 + np.random.normal(0, 0.02, len(df)))
# benchmark_df = pd.DataFrame({
#     'Date': df.index,
#     'NAV': benchmark_navs
# })
# benchmark_df.set_index('Date', inplace=True)
# benchmark_df['Benchmark_Return'] = benchmark_df['NAV'].pct_change()

# Prepare the data for LSTM
navs = df['NAV'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
navs_scaled = scaler.fit_transform(navs)
sequence_length = 5
X, y = create_sequences(navs_scaled, sequence_length)

# Split data into train and test sets
train_size = len(X) - 10
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Build and train the LSTM model
model = build_and_train_lstm(X_train, y_train)

# Predict on training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Inverse transform predictions
y_train_true = scaler.inverse_transform(y_train)
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_true = scaler.inverse_transform(y_test)
y_test_pred = scaler.inverse_transform(y_test_pred)

# Calculate MSE
train_mse = mean_squared_error(y_train_true, y_train_pred)
test_mse = mean_squared_error(y_test_true, y_test_pred)

print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")

# Forecast the next variable number of months (default to 6)
num_months = 6
last_sequence = navs_scaled[-sequence_length:]
forecast = forecast_nav(model, last_sequence, num_months)

# Inverse transform forecast
forecast = scaler.inverse_transform(forecast)

# Calculate average forecast for the given number of months
average_forecast = forecast.mean()

# Print results
print(f"Average forecast for the next {num_months} months: {average_forecast}")

# Calculate and print other financial features
features = calculate_features(df, benchmark_df)
print(f"Variance of monthly returns: {features['variance_monthly_returns']}")
print(f"Maximum drawdown: {features['max_drawdown']}")
print(f"Drawdown duration: {features['drawdown_duration']}")
print(f"Sharpe Ratio: {features['sharpe_ratio']}")
print(f"Sortino Ratio: {features['sortino_ratio']}")
print(f"Beta: {features['beta']}")
print(f"Expense Ratio: {features['expense_ratio']}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index, navs, label='Actual NAV')
forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=num_months, freq='M')
plt.plot(forecast_dates, forecast, label='Forecasted NAV', color='orange')
plt.legend()
plt.xlabel('Date')
plt.ylabel('NAV')
plt.title('NAV Forecasting with LSTM')
plt.show()
