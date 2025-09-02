from typing import TypedDict, Annotated
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os

# Load environment variables
load_dotenv()

# Debug print for API key
api_key = os.getenv("GROQ_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"API Key first 5 chars: {api_key[:5] if api_key else 'Not available'}")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    # Explicitly pass API key to ensure it's loaded
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
    print("ChatGroq initialized successfully")
except Exception as e:
    print(f"Error initializing ChatGroq: {e}")

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

# Compile the graph but store it in a different variable
graph_app = graph.compile()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "ChatBot API is running"}
    
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        print(f"Received message: {req.message}")
        # Use graph_app instead of app for invoking the graph
        result = graph_app.invoke({
            "messages": [HumanMessage(content=req.message)]
        })
        response = result["messages"][-1].content
        print(f"Generated response: {response[:50]}...")  # Print first 50 chars of response
        return {"response": response}
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"response": f"Sorry, I encountered an error: {str(e)}"}

# Add this at the bottom of the file to run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server at http://localhost:5000")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
