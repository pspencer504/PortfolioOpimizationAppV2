import sys
import os

# Ensure required packages are installed
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import streamlit as st
except ImportError:
    os.system(f"{sys.executable} -m pip install yfinance pandas numpy matplotlib scipy streamlit")
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import streamlit as st

# Streamlit App Title
st.title("Portfolio Optimization")

# Background Styling
st.markdown(
    """
    <style>
        body {
            background: url('https://source.unsplash.com/1600x900/?finance,technology') no-repeat center center fixed;
            background-size: cover;
            color: white;
            text-align: center;
        }
        .main-title {
            font-size: 48px;
            font-weight: bold;
            color: #00FFAA;
        }
        .sub-text {
            font-size: 24px;
            color: #FFD700;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Image Header
st.image("https://source.unsplash.com/800x400/?stocks,market", use_container_width=True)

# Sidebar Inputs
st.sidebar.header("User Inputs")
principal = st.sidebar.number_input("Enter Principal Capital ($)", min_value=0.0, value=1000.0, step=100.0)
tickers = st.sidebar.text_area("Enter Stock Tickers (comma-separated)").upper()
optimize_button = st.sidebar.button("Optimize Portfolio")

# Optimization Method Selection
st.header("1. Portfolio Input")
st.write("Enter the stocks you want to optimize and set your principal capital.")

st.header("2. Optimization Method")
optimization_method = st.selectbox("Choose Optimization Method:", [
    "Markowitz Portfolio Theory",
    "Efficient Frontier",
    "Monte Carlo Simulation",
    "Gaussian Copula Simulation",
    "Auto-regression Gaussian Copula Simulation"
])

st.header("3. Simulation Settings")
time_period = st.slider("Select historical time period (years):", 1, 10, 1)
prediction_period = st.slider("Select prediction time period (months):", 1, 24, 1)

# Run Optimization
if optimize_button and tickers:
    stock_list = [ticker.strip() for ticker in tickers.split(",")]

    # Fetch historical data
    st.write("### Fetching Data...")
    try:
        data = yf.download(stock_list, period=f"{time_period}y")['Close']
        returns = data.pct_change().dropna()

        # Fetch risk-free rate (10-year treasury yield)
        treasury_yield = yf.Ticker("^TNX")
        treasury_data = treasury_yield.history(period="1d")
        risk_free_rate = treasury_data['Close'].iloc[-1] / 100 if not treasury_data.empty else 0.02  # Default to 2%

        # Calculate Annualized Returns & Volatility
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)

        # Portfolio Performance Calculation
        def portfolio_performance(weights, returns):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            return portfolio_return, portfolio_volatility

        # Objective Function (Negative Sharpe Ratio)
        def negative_sharpe_ratio(weights, returns, risk_free_rate):
            portfolio_return, portfolio_volatility = portfolio_performance(weights, returns)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio

        # Portfolio Optimization
        num_assets = len(stock_list)
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))

        result = minimize(negative_sharpe_ratio, weights, args=(returns, risk_free_rate),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        # Optimized Portfolio Weights
        optimized_weights = result.x
        optimized_return, optimized_volatility = portfolio_performance(optimized_weights, returns)
        optimized_sharpe_ratio = (optimized_return - risk_free_rate) / optimized_volatility

        # Display Results
        st.success("Portfolio Optimization Complete!")
        st.write(f"**Optimized Expected Annual Return:** {optimized_return:.2%}")
        st.write(f"**Optimized Portfolio Volatility:** {optimized_volatility:.2%}")
        st.write(f"**Optimized Sharpe Ratio:** {optimized_sharpe_ratio:.2f}")

        # Show optimized weights
        st.write("### Optimized Portfolio Weights:")
        for i, weight in enumerate(optimized_weights):
            st.write(f"**{stock_list[i]}:** {weight:.2%}")

        # Plot Portfolio Allocation
        fig, ax = plt.subplots()
        ax.pie(optimized_weights, labels=stock_list, autopct="%1.1f%%", startangle=140)
        ax.axis("equal")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")

st.write("---")
st.write("Developed by: Paige Spencer, Ian Ortega, Nabil Othman, Chris Giamis")
