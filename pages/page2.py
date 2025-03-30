import streamlit as st

st.set_page_config(page_title="Page 2", page_icon="🌟")

st.title("This is Page 2 🌟")
st.write("You've successfully navigated to the second page!")

# Example chart
import pandas as pd
import numpy as np
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
st.line_chart(chart_data)

# Link back to main page
st.page_link("hello.py", label="← Go back to Home", icon="⬅️")