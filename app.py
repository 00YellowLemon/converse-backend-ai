import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage
from assistant import initialize_assistant
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize the assistant chain
try:
    assistant_chain = initialize_assistant()
except Exception as e:
    print(f"Error initializing assistant chain: {e}")
    assistant_chain = None

# FastAPI app
app = FastAPI(
    title="AI Conversation Coach API",
    description="Provides AI-powered feedback on user communication.",
    version="1.1.1"
)

# Add CORS middleware to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models (HistoryMessage remains, CoachingResponse is removed)
class HistoryMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'ai', 'other'")
    content: str = Field(..., description="Message content")

class CoachingRequest(BaseModel):
    user_input: str = Field(..., description="User's latest message/question.")
    dialogue_history: List[HistoryMessage] = Field(default=[], description="User <-> Other Party history.")
    coaching_history: List[HistoryMessage] = Field(default=[], description="User <-> AI Coach history.")

# CoachingResponse class has been removed

# Note: response_model removed from the decorator
@app.post("/coach")
async def handle_coaching_request(request: CoachingRequest) -> Dict[str, str]: # Return type hint changed
    """
    Processes user input and histories, calls the assistant chain,
    and returns coaching feedback as a dictionary.
    """
    if assistant_chain is None:
        raise HTTPException(status_code=503, detail="AI Coach service is unavailable.")

    coaching_history_messages = [HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content) for msg in request.coaching_history]
    dialogue_history_messages = [HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content) for msg in request.dialogue_history]

    input_data = {
        "user_input": request.user_input,
        "coaching_history": coaching_history_messages,
        "dialogue_history": dialogue_history_messages
    }

    try:
        ai_feedback = await assistant_chain.ainvoke(input_data)
        # Return a dictionary directly instead of a CoachingResponse instance
        return {"feedback": ai_feedback}
    except Exception as e:
        print(f"Error during assistant invocation: {e}")
        raise HTTPException(status_code=500, detail="Error processing request.")

@app.get("/")
async def root():
    return {"message": "AI Conversation Coach API is running!"}

if __name__ == "__main__":
    # Removed the initial import uvicorn as it's only needed here
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
