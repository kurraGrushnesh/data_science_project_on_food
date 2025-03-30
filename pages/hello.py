import streamlit as st

st.set_page_config(page_title="Hello", page_icon="👋")

st.title("Welcome to My App! 👋")
st.write("This is the main page of your Streamlit application.")

# Example interactive widget
name = st.text_input("What's your name?")
if name:
    st.write(f"Hello, {name}! Nice to meet you.")

# Link to other pages
st.page_link("pages/page2.py", label="Go to Page 2 →", icon="➡️")
