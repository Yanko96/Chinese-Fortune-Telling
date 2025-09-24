# app/fortune_api_utils.py
"""
API utilities for the Fortune Teller frontend.
"""

import requests
import streamlit as st
from datetime import datetime

import os

API_URL = "http://api.fortune.local:8000"

def get_fortune_response(question, session_id, model, query_type="general", birth_date=None, birth_gender=None, zodiac_sign=None):
    """
    Send a fortune telling request to the backend API.
    
    Args:
        question: The user's question
        session_id: Current session ID
        model: LLM model to use
        query_type: Type of fortune telling request (general, bazi, forecast)
        birth_date: Birth date for BaZi analysis
        birth_gender: Gender for BaZi analysis
        zodiac_sign: Chinese zodiac sign for forecasts
    
    Returns:
        API response or None if the request failed
    """
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    data = {
        "question": question,
        "model": model,
        "query_type": query_type
    }
    
    if session_id:
        data["session_id"] = session_id
    
    if birth_date:
        data["birth_date"] = birth_date
    
    if birth_gender:
        data["birth_gender"] = birth_gender
    
    if zodiac_sign:
        data["zodiac_sign"] = zodiac_sign

    try:
        response = requests.post(f"{API_URL}/fortune", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def get_zodiac_signs():
    """Get the list of Chinese zodiac signs from the API."""
    try:
        response = requests.get(f"{API_URL}/zodiac-signs")
        if response.status_code == 200:
            return response.json()
        else:
            return [
                "Rat (鼠)", "Ox (牛)", "Tiger (虎)", "Rabbit (兔)",
                "Dragon (龙)", "Snake (蛇)", "Horse (马)", "Goat (羊)",
                "Monkey (猴)", "Rooster (鸡)", "Dog (狗)", "Pig (猪)"
            ]
    except Exception:
        # Fallback if API is unavailable
        return [
            "Rat (鼠)", "Ox (牛)", "Tiger (虎)", "Rabbit (兔)",
            "Dragon (龙)", "Snake (蛇)", "Horse (马)", "Goat (羊)",
            "Monkey (猴)", "Rooster (鸡)", "Dog (狗)", "Pig (猪)"
        ]

def get_fortune_methods():
    """Get the list of fortune telling methods from the API."""
    try:
        response = requests.get(f"{API_URL}/fortune-methods")
        if response.status_code == 200:
            return response.json()
        else:
            # Fallback if API is unavailable
            return [
                {
                    "name": "BaZi (八字)",
                    "description": "Four Pillars of Destiny based on birth time",
                    "query_type": "bazi"
                },
                {
                    "name": "Yearly Forecast (年运)",
                    "description": "Predictions for the coming year based on zodiac sign",
                    "query_type": "forecast"
                },
                {
                    "name": "General Fortune (运势)",
                    "description": "General fortune telling and advice",
                    "query_type": "general"
                }
            ]
    except Exception:
        # Fallback if API is unavailable
        return [
            {
                "name": "BaZi (八字)",
                "description": "Four Pillars of Destiny based on birth time",
                "query_type": "bazi"
            },
            {
                "name": "Yearly Forecast (年运)",
                "description": "Predictions for the coming year based on zodiac sign",
                "query_type": "forecast"
            },
            {
                "name": "General Fortune (运势)",
                "description": "General fortune telling and advice",
                "query_type": "general"
            }
        ]

def calculate_chinese_zodiac(birth_year):
    """Calculate Chinese zodiac sign based on birth year."""
    zodiac_animals = [
        "Rat (鼠)", "Ox (牛)", "Tiger (虎)", "Rabbit (兔)",
        "Dragon (龙)", "Snake (蛇)", "Horse (马)", "Goat (羊)",
        "Monkey (猴)", "Rooster (鸡)", "Dog (狗)", "Pig (猪)"
    ]
    
    # Chinese zodiac cycle is 12 years, starting from Rat
    # Reference year 2020 is Rat
    index = (birth_year - 2020) % 12
    if index < 0:
        index += 12
    
    return zodiac_animals[index]