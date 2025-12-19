from fastapi import FastAPI
from pydantic import BaseModel
from graph import app_graph
from langchain_core.messages import HumanMessage

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def chat(req: ChatRequest):
    result = app_graph.invoke({
        "messages": [HumanMessage(content=req.message)]
    })

    return {
        "reply": result["messages"][-1].content
    }