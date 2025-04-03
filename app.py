import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from assistant import initialize_assistant

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

class HistoryMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'ai', 'other'")
    content: str = Field(..., description="Message content")

class CoachingRequest(BaseModel):
    user_input: str = Field(..., description="User's latest message/question.")
    dialogue_history: List[HistoryMessage] = Field(default=[], description="User <-> Other Party history.")
    coaching_history: List[HistoryMessage] = Field(default=[], description="User <-> AI Coach history.")

@app.post("/coach")
async def handle_coaching_request(request: CoachingRequest) -> Dict[str, str]:
    """
    Processes user input and histories, returns coaching feedback as a dictionary.
    """
    if assistant_chain is None:
        raise HTTPException(status_code=503, detail="AI Coach service is unavailable.")

    input_data = {
        "user_input": request.user_input,
        "coaching_history": request.coaching_history,
        "dialogue_history": request.dialogue_history
    }

    try:
        ai_feedback = await assistant_chain.ainvoke(input_data)
        return {"feedback": ai_feedback}
    except Exception as e:
        print(f"Error during assistant invocation: {e}")
        raise HTTPException(status_code=500, detail="Error processing request.")

@app.get("/")
async def root():
    return {"message": "AI Conversation Coach API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)