from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

load_dotenv()

model = init_chat_model("anthropic:claude-sonnet-4.5")

memory = MemorySaver()

class State(BaseModel):
    """messages have a type of list"""
    """When updating the key state, it should be append the new messages to the list, rather than overwrite them"""
    messages: Annotated[list, add_messages]

def superbot(state: State):
    return {"messages": [model.invoke(state.messages)]}

graph_builder = StateGraph(State)

graph_builder.add_node("super_bot", superbot)

graph_builder.add_edge(START, "super_bot")
graph_builder.add_edge("super_bot", END)

graph = graph_builder.compile(checkpointer=memory)

config: RunnableConfig = {"configurable": {"thread_id": "user-1"}}

for chunk in graph.stream(State(messages = [HumanMessage(content="Hi, my name is John and i like art")]), config, stream_mode="updates"): #AI message only
    print(chunk)

for chunk in graph.stream(State(messages = [HumanMessage(content="Hi, my name is John and i like art")]), config, stream_mode="values"): #AI and human message
    print(chunk)

for chunk in graph.stream(State(messages = [HumanMessage(content="Hi, my name is John and i like art")]), config, version="v2"): #Way more data would be added in the output
    print(chunk)