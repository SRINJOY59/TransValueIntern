import os

# Directory where the comparison files will be created
folder_name = "pages"
os.makedirs(folder_name, exist_ok=True)

# Relevant comparisons based on the dataset columns
relevant_comparisons = [
    ("4_month_rolling_return", "monthly_return"),
    ("variance_of_monthly_returns", "sharpe_ratio"),
    ("CAGR", "expense_ratio"),
    ("max_drawdown", "variance_of_monthly_returns"),
    ("monthly_return", "sharpe_ratio"),
    ("CAGR", "4_month_rolling_return"),
    ("max_drawdown", "CAGR")
]

# Template for each comparison page
template = """
import streamlit as st
import pandas as pd


# Criteria based on the actual dataset columns
criteria = [
    "4_month_rolling_return", 
    "monthly_return", 
    "variance_of_monthly_returns", 
    "sharpe_ratio", 
    "CAGR", 
    "max_drawdown"
]

# Relevant comparison for this page
comparison_index = {index}
crit_1, crit_2 = {comparison}

st.title(f"Comparison {{index + 1}}")

def create_comparison_slider(crit_1, crit_2):
    st.header(f"Compare {{crit_1}} vs {{crit_2}}")
    comparison_value = st.slider(f"How much more important is {{crit_1}} compared to {{crit_2}}?", 
                                 1.0, 9.0, 1.0)
    return comparison_value

comparison_value = create_comparison_slider(crit_1, crit_2)
st.write(f"Comparison Value for {{crit_1}} vs {{crit_2}}: {{comparison_value}}")

if "comparisons_made" not in st.session_state:
    st.session_state.comparisons_made = []

if st.button("Submit"):
    st.session_state.comparisons_made.append((crit_1, crit_2, comparison_value))
    
    # Redirect to next comparison page
    if comparison_index < 7:
        next_page = f"Comparison{{comparison_index + 2}}"
        st.session_state.current_page = next_page
        st.experimental_set_query_params(page=next_page)
    else:
        st.session_state.current_page = "Results"
        st.experimental_set_query_params(page="Results")
"""

# Generate the comparison files
for i, comparison in enumerate(relevant_comparisons):
    filename = f"Comparison{i + 1}.py"
    file_path = os.path.join(folder_name, filename)
    
    # Prepare the comparison string in a format-compatible way
    comparison_str = str(comparison)
    
    with open(file_path, "w") as file:
        content = template.format(index=i, comparison=comparison_str)
        file.write(content)

print(f"Comparison files created in the '{folder_name}' folder.")
