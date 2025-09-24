# api/fortune_main.py
"""
API endpoints for the Chinese Fortune Teller application.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import uuid
import logging
import re
from datetime import datetime

from db_utils import insert_application_logs, get_chat_history
from fortune_langchain_utils import get_fortune_chain

import os
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")

# Set up logging
logging.basicConfig(filename='fortune_app.log', level=logging.INFO)
app = FastAPI()

# Enums and Models
class ModelName(str, Enum):
    GEMINI_FLASH = "gemini-2.0-flash-exp"
    GEMINI_PRO = "gemini-2.0-pro-exp"

class QueryType(str, Enum):
    GENERAL = "general"
    BAZI = "bazi"
    FORECAST = "forecast"

class FortuneInput(BaseModel):
    question: str
    session_id: Optional[str] = Field(default=None)
    model: ModelName = Field(default=ModelName.GEMINI_PRO)
    query_type: QueryType = Field(default=QueryType.GENERAL)
    birth_date: Optional[str] = Field(default=None)  # Format: YYYY-MM-DD HH:MM
    birth_gender: Optional[str] = Field(default=None)  # "male" or "female"
    zodiac_sign: Optional[str] = Field(default=None)

class FortuneResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    query_type: QueryType

# Utility function to validate birth date
def validate_birth_date(birth_date: str) -> bool:
    if not birth_date:
        return False
    
    # Check format YYYY-MM-DD HH:MM or YYYY-MM-DD
    date_pattern = r'^\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?$'
    if not re.match(date_pattern, birth_date):
        return False
    
    # Additional validation could be added here
    return True

# API Endpoints
@app.post("/fortune", response_model=FortuneResponse)
def get_fortune(fortune_input: FortuneInput):
    # Initialize session if needed
    session_id = fortune_input.session_id or str(uuid.uuid4())
    
    # Log the request
    logging.info(
        f"Session ID: {session_id}, Query Type: {fortune_input.query_type.value}, "
        f"Question: {fortune_input.question}, Model: {fortune_input.model.value}"
    )
    
    # Validate inputs for specific query types
    if fortune_input.query_type == QueryType.BAZI and not validate_birth_date(fortune_input.birth_date):
        raise HTTPException(
            status_code=400, 
            detail="Valid birth date (YYYY-MM-DD HH:MM) is required for BaZi analysis"
        )
    
    # Get chat history
    chat_history = get_chat_history(session_id)
    
    # Process input based on query type
    if fortune_input.query_type == QueryType.BAZI:
        # For BaZi analysis, we add birth date to the question
        question = (
            f"BaZi analysis for someone born on {fortune_input.birth_date}, "
            f"gender: {fortune_input.birth_gender or 'not specified'}. {fortune_input.question}"
        )
    elif fortune_input.query_type == QueryType.FORECAST:
        # For yearly forecast
        current_year = datetime.now().year
        question = (
            f"Yearly forecast for {current_year} for {fortune_input.zodiac_sign or 'a person'} "
            f"with the question: {fortune_input.question}"
        )
    else:
        # General fortune telling question
        question = fortune_input.question
    
    # Get the appropriate chain
    fortune_chain = get_fortune_chain(
        query_type=fortune_input.query_type.value,
        model=fortune_input.model.value
    )
    
    # Generate the fortune
    fortune_result = fortune_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    
    # Extract the answer
    answer = fortune_result.get('answer', "I cannot see clearly at this moment. Please ask again.")
    
    # Log the response
    insert_application_logs(session_id, question, answer, fortune_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer[:100]}...")
    
    # Return the response
    return FortuneResponse(
        answer=answer,
        session_id=session_id,
        model=fortune_input.model,
        query_type=fortune_input.query_type
    )

@app.get("/zodiac-signs", response_model=List[str])
def get_zodiac_signs():
    """Returns a list of the 12 Chinese zodiac signs."""
    return [
        "Rat (鼠)", "Ox (牛)", "Tiger (虎)", "Rabbit (兔)",
        "Dragon (龙)", "Snake (蛇)", "Horse (马)", "Goat (羊)",
        "Monkey (猴)", "Rooster (鸡)", "Dog (狗)", "Pig (猪)"
    ]

@app.get("/fortune-methods", response_model=List[dict])
def get_fortune_methods():
    """Returns a list of fortune telling methods with descriptions."""
    return [
        {
            "name": "BaZi (八字)",
            "description": "Four Pillars of Destiny based on birth time",
            "query_type": "bazi"
        },
        {
            "name": "Zi Wei Dou Shu (紫微斗数)",
            "description": "Purple Star Astrology for detailed life readings",
            "query_type": "general"
        },
        {
            "name": "Qi Men Dun Jia (奇門遁甲)",
            "description": "Ancient divination system for strategic decisions",
            "query_type": "general"
        },
        {
            "name": "I Ching (易经)",
            "description": "Book of Changes divination using hexagrams",
            "query_type": "general"
        },
        {
            "name": "Yearly Forecast (年运)",
            "description": "Predictions for the coming year based on zodiac sign",
            "query_type": "forecast"
        }
    ]