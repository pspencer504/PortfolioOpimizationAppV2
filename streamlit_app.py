import streamlit as st 
import streamlit.components.v1 as components

# App Title
st.title("Portfolio Optimization")
#st.subheader("...")

# Background and Styling
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

# Graphic Placeholder
st.image("https://source.unsplash.com/800x400/?stocks,market", use_container_width=True)

# Navigation Button
if "page" not in st.session_state:
    if st.button("Optimize Investments Here"):
        st.session_state["page"] = "optimization"
        st.rerun()

# Optimization Page
if "page" in st.session_state and st.session_state["page"] == "optimization":
    st.sidebar.header("User Inputs")
    principal = st.sidebar.number_input("Enter Principal Capital ($)", min_value=0.0, value=1000.0, step=100.0)
    tickers = st.sidebar.text_area("Enter Stock Tickers (comma-separated)")
    optimize_button = st.sidebar.button("Optimize Portfolio")

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

    if optimize_button:
        st.write("### User Inputs:")
        st.write(f"**Principal Capital:** ${principal}")
        st.write(f"**Selected Tickers:** {tickers}")
        st.write(f"**Optimization Method:** {optimization_method}")
        st.write(f"**Historical Time Period:** {time_period} years")
        st.write(f"**Prediction Time Period:** {prediction_period} months")
        st.success("Portfolio optimization will be implemented here!")
st.write("---")
st.write("Developed by: Paige Spencer, Ian Ortega, Nabil Othman, Chris Giamis")"

