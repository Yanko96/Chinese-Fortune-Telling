# app/fortune_chat_interface.py
"""
Chat interface component for the Fortune Teller application.
"""

import streamlit as st
from fortune_api_utils import get_fortune_response

def display_fortune_chat():
    """Display the fortune telling chat interface."""
    # Apply custom styling to messages
    st.markdown("""
    <style>
    .fortune-header {
        text-align: center;
        color: #9c4046;
        font-family: 'Arial', sans-serif;
    }
    .fortune-subheader {
        text-align: center;
        color: #5f5f5f;
        font-style: italic;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Fortune Teller's introduction messages
    if not st.session_state.messages:
        # Add initial messages for first-time users
        st.session_state.messages.append({
            "role": "assistant",
            "content": "欢迎来到果赖算命阁！我是果赖，精通八字、紫微斗数、奇门遁甲及易经。告诉我你的问题，我将为你揭示天机..."
        })
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Welcome to Master Guo Lai's Fortune Hall! I am Guo Lai, master of BaZi, Zi Wei Dou Shu, Qi Men Dun Jia, and the I Ching. Share your questions, and I shall reveal the secrets of fate..."
        })

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Add special styling for fortune teller messages
            if message["role"] == "assistant":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your destiny..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.spinner("果赖正在推算命理..."):
            # Get relevant parameters based on query type
            birth_date = st.session_state.get("birth_datetime") if st.session_state.get("query_type") == "bazi" else None
            birth_gender = st.session_state.get("birth_gender") if st.session_state.get("query_type") == "bazi" else None
            zodiac_sign = st.session_state.get("zodiac_sign") if st.session_state.get("query_type") in ["bazi", "forecast"] else None
            
            # Get response from API
            response = get_fortune_response(
                prompt, 
                st.session_state.session_id, 
                st.session_state.model,
                query_type=st.session_state.query_type,
                birth_date=birth_date,
                birth_gender=birth_gender,
                zodiac_sign=zodiac_sign
            )
            
            if response:
                st.session_state.session_id = response.get('session_id')
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                
                with st.chat_message("assistant"):
                    st.markdown(response['answer'])
                    
                    # Optional expandable details
                    with st.expander("占卜详情 (Divination Details)"):
                        st.subheader("占卜方法 (Method)")
                        st.code(response['query_type'])
                        st.subheader("神算引擎 (Oracle Engine)")
                        st.code(response['model'])
            else:
                st.error("无法连接天机... (Unable to connect to the cosmic forces. Please try again.)")