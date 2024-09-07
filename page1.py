import streamlit as st
import google.generativeai as genai

GOOGLE_API_KEY = 'AIzaSyDqwaZbyXjUmXkq8-yG3Ay91h87sv8jkSg'
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.0-pro-latest')

def generate_fund_recommendations(risk, max_drawdown, beta, 
                                  rolling_return_fund_size, expense_ratio, drawdown_duration, sortino_ratio):
    prompt = f"Suggest some top mutual funds based on these values of user requirements: \
               \nVariance of Features (Risk): {risk}, \
               \nMax Drawdown: {max_drawdown}%, \
               \nBeta: {beta}, \
               \nRolling Return: {rolling_return_fund_size}%, \
               \nExpense Ratio: {expense_ratio}%, \
               \nDrawdown Duration: {drawdown_duration} months, \
               \nSortino Ratio: {sortino_ratio}."

    response = model.generate_content(prompt)
    return response.text

def page1():
    st.title("Mutual Fund Suggestions")

    risk = st.session_state.risk
    max_drawdown = st.session_state.max_drawdown
    beta = st.session_state.beta
    rolling_return_fund_size = st.session_state.rolling_return_fund_size
    expense_ratio = st.session_state.expense_ratio
    drawdown_duration = st.session_state.drawdown_duration
    sortino_ratio = st.session_state.sortino_ratio

    recommendations = generate_fund_recommendations(
        risk, max_drawdown, beta, 
        rolling_return_fund_size, expense_ratio, drawdown_duration, sortino_ratio
    )

    st.subheader("Top Mutual Fund Recommendations:")
    st.write(recommendations)

