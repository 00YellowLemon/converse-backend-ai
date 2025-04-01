import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

def initialize_assistant():
    """Initializes and returns the LangChain coaching chain."""
    # --- Configuration ---
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    try:
        # --- LangChain Setup ---
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                # ... (Keep the same detailed system prompt) ...
                "You are an AI conversation coach. You have access to two distinct histories: "
                "1. 'coaching_history': Your past interactions and feedback provided to the 'user'. Use this to maintain context, avoid repetition, and track progress. "
                "2. 'dialogue_history': The conversation the 'user' is having with another party. This is the primary subject of your analysis. "
                "Your main task is to analyze the latest 'user_input' (which could be the user's message to the other party OR a question directed at you, the coach) "
                "within the context of the 'dialogue_history'. "
                "Leverage the 'coaching_history' for continuity in your coaching style and advice. "
                "Provide constructive feedback DIRECTLY TO THE USER on their communication within the 'dialogue_history' (focusing on emotional intelligence, clarity, conflict resolution). "
                "If the 'user_input' is a question for you, answer it helpfully, using both histories if relevant."
            )),
            MessagesPlaceholder(variable_name="coaching_history"),
            MessagesPlaceholder(variable_name="dialogue_history"),
            HumanMessage(content="{user_input}")
        ])

        output_parser = StrOutputParser()

        chain = prompt | model | output_parser
        print("AI Assistant Chain Initialized Successfully.") # Add confirmation
        return chain

    except Exception as e:
        print(f"FATAL ERROR: Could not initialize LangChain components in assistant.py: {e}")
        # Re-raise or return None to signal failure
        raise RuntimeError(f"Failed to initialize assistant: {e}") from e