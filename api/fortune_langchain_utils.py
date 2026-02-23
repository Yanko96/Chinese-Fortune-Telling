# api/fortune_langchain_utils.py
"""
LangChain utilities specialized for the Chinese Fortune Teller application.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os

from chroma_utils import get_vectorstore
from fortune_prompts import (
    fortune_contextualize_prompt, 
    fortune_qa_prompt,
    birthday_analysis_prompt,
    yearly_forecast_prompt
)

# Set up retriever
retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})
output_parser = StrOutputParser()

def get_fortune_chain(query_type="general", model="gemini-2.5-flash"):
    """
    Creates a specialized RAG chain for fortune telling.
    
    Args:
        query_type: Type of fortune telling query
            - "general": General fortune telling questions
            - "bazi": BaZi analysis based on birth date
            - "forecast": Yearly forecasts
        model: LLM model to use
    
    Returns:
        A retrieval chain configured for the specified query type
    """
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.7,  # Slightly higher temperature for creative fortune telling responses
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Set up the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, 
        fortune_contextualize_prompt
    )
    
    # Select the appropriate QA prompt based on query type
    if query_type == "bazi":
        qa_prompt = birthday_analysis_prompt
    elif query_type == "forecast":
        qa_prompt = yearly_forecast_prompt
    else:  # general
        qa_prompt = fortune_qa_prompt
    
    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create and return the full RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain