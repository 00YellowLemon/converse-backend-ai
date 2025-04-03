import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def format_history(history):
    """Formats a list of messages into a string conversation."""
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

def initialize_assistant():
    """Initializes and returns the LangChain coaching chain."""
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

        prompt_template = """
        You are an AI conversation coach. Analyze the following conversation and provide feedback:

        Coaching History (your past interactions with the user):
        {coaching_history}

        Dialogue History (conversation between user and another person):
        {dialogue_history}

        User's latest input: {user_input}

        Provide constructive feedback on the user's communication, focusing on:
        - Emotional intelligence
        - Clarity
        - Conflict resolution
        - Relationship building
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        chain = (
            prompt 
            | model 
            | StrOutputParser()
        )
        
        print("AI Assistant Chain Initialized Successfully.")
        return chain

    except Exception as e:
        print(f"FATAL ERROR: Could not initialize LangChain components: {e}")
        raise RuntimeError(f"Failed to initialize assistant: {e}") from e