import streamlit as st

# Check if the current page is set in session state, default to 1
if "current_page" not in st.session_state:
    st.session_state.current_page = "Comparison1"

# Display the home page content
st.title("Welcome to the Ultimate Mutual Fund Selector")

st.write("""
Discover the best mutual funds tailored to your preferences! Through a series of intuitive comparisons, we'll guide you to make the smartest investment choices using the powerful Analytical Hierarchy Process (AHP). Ready to begin your journey? Let's dive in!
""")

if st.button("Start Comparisons"):
    st.session_state.current_page = "Comparison1"
    st.experimental_set_query_params(page="Comparison1")

