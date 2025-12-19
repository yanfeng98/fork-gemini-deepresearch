import os
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain.chat_models import init_chat_model

class GraphState(TypedDict):
    messages: list

llm = init_chat_model(
    model="openai:deepseek-v3-2-251201",
    temperature=0.0,
    streaming=True,
    base_url=os.environ.get("DEEPSEEK_API_BASE"),
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)

def chat_node(state: GraphState):
    response = llm.invoke(state["messages"])
    return {
        "messages": state["messages"] + [response]
    }

graph = StateGraph(GraphState)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.set_finish_point("chat")

app_graph = graph.compile()