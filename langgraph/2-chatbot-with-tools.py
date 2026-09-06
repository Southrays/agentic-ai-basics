from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

load_dotenv()

model = init_chat_model("anthropic:claude-sonnet-4.5")

tool = TavilySearch(max_results=2)

class State(BaseModel):
    """messages have a type of list"""
    """When updating the key state, it should be append the new messages to the list, rather than overwrite them"""
    messages: Annotated[list, add_messages]

def multiply(a: int, b: int) -> int:
    """This function multiplies a by b"""
    return a * b

tools = [tool, multiply]

model_with_tool = model.bind_tools(tools)

def tool_calling_model(state: State):
    return {"messages": [model_with_tool.invoke(state.messages)]}

graph_builder = StateGraph(State)

graph_builder.add_node("tool_calling_model", tool_calling_model)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "tool_calling_model")
graph_builder.add_conditional_edges(
    "tool_calling_model", 
    tools_condition
)
graph_builder.add_edge("tools", "tool_calling_model")

graph = graph_builder.compile()

response = graph.invoke(State(messages = [HumanMessage(content="What is the latest news in ai?")]))
print(response["messages"][-1])

second_response = graph.invoke(State(messages = [HumanMessage(content="What is 2 * 3")]))
print(second_response["messages"][-1])