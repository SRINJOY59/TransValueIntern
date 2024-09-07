import streamlit as st
from page1 import page1
from page2 import page2

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;  /* Light grayish-blue background */
    }
    .stApp div {
        color: #1f2c56; /* Dark blue text color for better contrast */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
        
    if 'home_submitted' not in st.session_state:
        st.session_state.home_submitted = False

    def navigate_to(page):
        st.session_state.page = page

    if st.session_state.page == 'Home':
        st.title("Mutual Fund Performance Evaluation")
        st.subheader("Predictive Scores of Selected Mutual Funds:")

        funds = {
            "Vanguard 500 Index Fund": 0.92,
            "Fidelity Contrafund": 0.88,
            "T. Rowe Price Blue Chip Growth Fund": 0.90,
            "American Funds Growth Fund of America": 0.85,
            "Invesco QQQ Trust": 0.91
        }

        for fund, score in funds.items():
            st.write(f"{fund}: {score}")

        st.subheader("Please select a mutual fund scheme:")

        selected_fund = st.selectbox(
            "Select a Mutual Fund Scheme",
            list(funds.keys())
        )

        st.write(f"You selected: **{selected_fund}**")

        st.subheader("Please customize the following metrics for the selected scheme (if needed):")

        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Customize Variance of Features (Risk)"):
                st.session_state.risk = st.slider(
                    "Variance of Features (Risk)", 0.0, 0.5, 0.1, step=0.01, help="The variance of features indicating the risk level."
                )
            else:
                st.session_state.risk = 0.1

            if st.checkbox("Customize Max Drawdown (%)"):
                st.session_state.max_drawdown = st.slider(
                    "Max Drawdown (%)", 0, 50, 10, help="The maximum observed loss from a peak to a trough before a new peak."
                )
            else:
                st.session_state.max_drawdown = 10

            if st.checkbox("Customize Beta"):
                st.session_state.beta = st.slider(
                    "Beta", -1.0, 2.0, 1.0, step=0.01, help="A measure of the volatility of the fund relative to the market."
                )
            else:
                st.session_state.beta = 1.0
        
        with col2:
            if st.checkbox("Customize Rolling Return (%)"):
                st.session_state.rolling_return_fund_size = st.slider(
                    "Rolling Return (%)", 0.0, 20.0, 5.0, step=0.1, help="The fund's average rolling return over a specified period."
                )
            else:
                st.session_state.rolling_return_fund_size = 5.0

            if st.checkbox("Customize Expense Ratio (%)"):
                st.session_state.expense_ratio = st.slider(
                    "Expense Ratio (%)", 0.0, 2.0, 0.5, step=0.01, help="The annual fee expressed as a percentage of the fund's assets."
                )
            else:
                st.session_state.expense_ratio = 0.5

            if st.checkbox("Customize Drawdown Duration (Months)"):
                st.session_state.drawdown_duration = st.slider(
                    "Drawdown Duration (Months)", 0, 24, 6, help="The time it takes for the fund to recover from a drawdown."
                )
            else:
                st.session_state.drawdown_duration = 6

            if st.checkbox("Customize Sortino Ratio"):
                st.session_state.sortino_ratio = st.slider(
                    "Sortino Ratio", 0.0, 3.0, 1.0, step=0.01, help="A measure of risk-adjusted return that focuses on downside risk."
                )
            else:
                st.session_state.sortino_ratio = 1.0
            
        if st.button("Submit"):
            st.session_state.home_submitted = True
            st.session_state.selected_fund = selected_fund  
            navigate_to('Page 1')

    elif st.session_state.page == 'Page 1':
        page1()
        if st.session_state.get('page1_submitted', False):
            if st.button("Go to Page 2"):
                navigate_to('Page 2')

    elif st.session_state.page == 'Page 2':
        page2()

if __name__ == "__main__":
    main()
