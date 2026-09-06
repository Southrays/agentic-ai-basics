from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from pydantic import BaseModel

load_dotenv()

model = init_chat_model("anthropic:claude-sonnet-4.5")

memory = MemorySaver()

tavily_tool = TavilySearch(max_results=2)

class State(BaseModel):
    """messages have a type of list"""
    """When updating the key state, it should be append the new messages to the list, rather than overwrite them"""
    messages: Annotated[list, add_messages]

def human_assistance(query: str) -> str:
    """Request assistance from a human"""
    human_response = interrupt({"query": query})
    return human_response["data"]

tools = [tavily_tool, human_assistance]

model_with_tools = model.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [model_with_tools.invoke(state.messages)]}

graph_builder = StateGraph(State)

graph_builder.add_node("chat_bot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chat_bot")
graph_builder.add_conditional_edges("chat_bot", tools_condition)
graph_builder.add_edge("tools", "chat_bot")

graph = graph_builder.compile(checkpointer=memory)

config: RunnableConfig = {"configurable": {"thread_id": "user-1"}}

user_input = "I need some expert guidiance and assistance for building an AI agent. Could you request assistance for me?"
response = graph.stream(State(messages=[HumanMessage(content=user_input)]), config, stream_mode="values")

for chunk in response:
    if "messages" in chunk:
        chunk["messages"][-1].pretty_print()


human_response = (
    "We the experts are here to help, we suggest you check out Langgraph to build your agent"
    "It's much more reliable and extensible than simple autonomous agents"
)

human_command = Command(resume={"data": human_response})
response2 = graph.stream(human_command, config, stream_mode="values")

for chunk in response2:
    if "messages" in chunk:
        chunk["messages"][-1].pretty_print()