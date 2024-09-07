import streamlit as st

# Define page options
PAGES = {
    "Criteria Selection": "criteria_selection.py",
    "Pairwise Comparisons": "pairwise_comparisons.py"
}

def main():
    st.title("Mutual Fund Selection using AHP")
    
    # Page selector
    selection = st.sidebar.radio("Choose a page", list(PAGES.keys()))
    selected_page = PAGES[selection]
    
    # Run the selected page
    with open(selected_page) as f:
        exec(f.read())

if __name__ == "__main__":
    main()
