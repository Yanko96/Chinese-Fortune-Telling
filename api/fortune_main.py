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
import sys
import re
from datetime import datetime
from contextlib import asynccontextmanager

from db_utils import insert_application_logs, get_chat_history, create_application_logs, create_document_store
from fortune_langchain_utils import get_fortune_chain

import os
os.environ["MOONSHOT_API_KEY"] = os.environ.get("MOONSHOT_API_KEY", "")

# Set up logging
# Configure logging to both file and stdout (so CloudWatch captures logs)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('fortune_app.log'),
        logging.StreamHandler(sys.stdout)
    ],
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)

# Support mounting the API behind a path prefix (e.g., "/api") when using an ALB
# Configure via env: API_ROOT_PATH=/api
API_ROOT_PATH = os.getenv("API_ROOT_PATH", "")
DEBUG_MODE = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite tables on startup
    create_application_logs()
    create_document_store()
    logging.info("Database tables initialized.")
    yield

app = FastAPI(root_path=API_ROOT_PATH, lifespan=lifespan)

# Enums and Models
class QueryType(str, Enum):
    GENERAL = "general"
    BAZI = "bazi"
    FORECAST = "forecast"

class FortuneInput(BaseModel):
    question: str
    session_id: Optional[str] = Field(default=None)
    model: str = Field(default="moonshot-v1-8k")
    query_type: QueryType = Field(default=QueryType.GENERAL)
    birth_date: Optional[str] = Field(default=None)  # Format: YYYY-MM-DD HH:MM
    birth_gender: Optional[str] = Field(default=None)  # "male" or "female"
    zodiac_sign: Optional[str] = Field(default=None)

class FortuneResponse(BaseModel):
    answer: str
    session_id: str
    model: str
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
@app.get("/healthz")
@app.get("/api/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/fortune", response_model=FortuneResponse)
@app.post("/api/fortune", response_model=FortuneResponse)
def get_fortune(fortune_input: FortuneInput):
    try:
        # Initialize session if needed
        session_id = fortune_input.session_id or str(uuid.uuid4())

        # Log the request
        logging.info(
            f"Session ID: {session_id}, Query Type: {fortune_input.query_type.value}, "
            f"Question: {fortune_input.question}, Model: {fortune_input.model}"
        )

        # Get chat history
        chat_history = get_chat_history(session_id)

        # ── Effective query_type resolution ────────────────────────────────
        # When the user sets a birthday in the sidebar, the frontend sends
        # query_type=bazi on every subsequent request — including chitchat
        # ("你好", "你能做什么"). The router's skip_rag branch correctly
        # avoids retrieval for those, but the BaZi-specialized prompt would
        # still force the LLM into "calculate Year Pillar / Day Master / ..."
        # mode. Detected this during self-testing — a greeting came back as
        # a full BaZi reading instead of a hello.
        #
        # Fix: mirror the router's skip_rag check at this layer. If the user
        # is just chatting, downgrade to GENERAL so the generic persona prompt
        # answers — the chain's retriever will also independently land on
        # skip_rag, so the LLM call doesn't fetch context either way.
        from retriever_router import should_skip_rag  # type: ignore
        skip, skip_reason = should_skip_rag(fortune_input.question)

        if skip:
            effective_query_type = QueryType.GENERAL
            question = fortune_input.question  # no English wrapper for chitchat
            logging.info(
                f"[api] effective_query_type=GENERAL (was {fortune_input.query_type.value}) "
                f"— skip_rag={skip_reason}"
            )
        else:
            effective_query_type = fortune_input.query_type
            if fortune_input.query_type == QueryType.BAZI:
                # Real BaZi request — validate birth_date here, not earlier,
                # so chitchat with query_type=bazi but no birth_date still works
                if not validate_birth_date(fortune_input.birth_date):
                    raise HTTPException(
                        status_code=400,
                        detail="Valid birth date (YYYY-MM-DD HH:MM) is required for BaZi analysis"
                    )
                question = (
                    f"BaZi analysis for someone born on {fortune_input.birth_date}, "
                    f"gender: {fortune_input.birth_gender or 'not specified'}. {fortune_input.question}"
                )
            elif fortune_input.query_type == QueryType.FORECAST:
                current_year = datetime.now().year
                question = (
                    f"Yearly forecast for {current_year} for {fortune_input.zodiac_sign or 'a person'} "
                    f"with the question: {fortune_input.question}"
                )
            else:
                question = fortune_input.question

        # Get the appropriate chain using the effective (post-override) type
        fortune_chain = get_fortune_chain(
            query_type=effective_query_type.value,
            model=fortune_input.model
        )

        # Generate the fortune
        fortune_result = fortune_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })

        # Extract the answer
        answer = fortune_result.get('answer', "I cannot see clearly at this moment. Please ask again.")

        # Log the response
        insert_application_logs(session_id, question, answer, fortune_input.model)
        logging.info(f"Session ID: {session_id}, AI Response: {answer[:100]}...")

        # Return the response. We surface the effective_query_type (not the
        # frontend-sent one) so clients can detect the skip_rag downgrade and
        # render accordingly — e.g. a Streamlit panel that says "BaZi reading"
        # should hide itself when the response was actually a chitchat reply.
        return FortuneResponse(
            answer=answer,
            session_id=session_id,
            model=fortune_input.model,
            query_type=effective_query_type
        )
    except HTTPException:
        # Propagate HTTP exceptions as-is
        raise
    except Exception as e:
        msg = str(e).lower()
        # Map upstream LLM rate-limit / overload (Kimi 429 + Moonshot's
        # 'engine_overloaded_error') to 503 so clients can distinguish
        # 'we're broken' from 'upstream is busy, retry shortly'.
        # langchain-openai already retries 2x internally via ChatOpenAI's
        # max_retries=2 — by the time we see the exception, retries are
        # exhausted, so falling further back at this layer is the right
        # signal to the caller.
        if any(kw in msg for kw in ("429", "rate limit", "overloaded", "engine_overloaded")):
            logging.warning(f"Upstream LLM rate-limited/overloaded: {e}")
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM 上游暂时拥堵，请几秒后重试。"
                    "(Upstream LLM is temporarily overloaded — please retry.)"
                ),
            )
        # Connection / timeout errors → 504 Gateway Timeout
        if any(kw in msg for kw in ("connection error", "timeout", "timed out")):
            logging.warning(f"Upstream LLM connection issue: {e}")
            raise HTTPException(
                status_code=504,
                detail="Upstream LLM service unreachable or slow. Please retry shortly.",
            )
        logging.exception("Unhandled exception in /fortune")
        # Only expose internal details when DEBUG is explicitly enabled
        if DEBUG_MODE:
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
        else:
            raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/zodiac-signs", response_model=List[str])
@app.get("/api/zodiac-signs", response_model=List[str])
def get_zodiac_signs():
    """Returns a list of the 12 Chinese zodiac signs."""
    return [
        "Rat (鼠)", "Ox (牛)", "Tiger (虎)", "Rabbit (兔)",
        "Dragon (龙)", "Snake (蛇)", "Horse (马)", "Goat (羊)",
        "Monkey (猴)", "Rooster (鸡)", "Dog (狗)", "Pig (猪)"
    ]

@app.get("/fortune-methods", response_model=List[dict])
@app.get("/api/fortune-methods", response_model=List[dict])
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