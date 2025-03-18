# app/fortune_app.py
"""
Main Streamlit application for the Chinese Fortune Teller.
"""

import streamlit as st
from fortune_sidebar import display_fortune_sidebar
from fortune_chat_interface import display_fortune_chat

# Set page configuration
st.set_page_config(
    page_title="师太算命阁 | Master Shi Tai's Fortune Hall",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f4e3;
        background-image: url('https://www.transparenttextures.com/patterns/rice-paper.png');
    }
    .stApp {
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #9c4046;
    }
    .stSidebar {
        background-color: #f0e9d2;
    }
    .css-1aumxhk {
        background-color: #f5deb3;
    }
</style>
""", unsafe_allow_html=True)

# Main header
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <h1 class="fortune-header">🏮 果赖算命阁 🏮</h1>
    <h3 class="fortune-subheader">Master Guo Lai's Fortune Hall</h3>
    """, unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "query_type" not in st.session_state:
    st.session_state.query_type = "general"

# Display the sidebar
display_fortune_sidebar()

# Display the chat interface
display_fortune_chat()

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; font-size: 0.8em;">
    本算命结果仅供参考，请谨慎决策。<br>
    These divinations are for entertainment purposes only. Please use wisdom in your decisions.
</div>
""", unsafe_allow_html=True)