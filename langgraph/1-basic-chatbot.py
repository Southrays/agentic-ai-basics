from typing import Annotated

from dotenv import load_dotenv

# from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("anthropic:claude-sonnet-4.0")  

class State(BaseModel):
    """messages have a type of list"""
    """When updating the key state, it should be append the new messages to the list, rather than overwrite them"""
    messages: Annotated[list, add_messages] = Field(description="The messages history")
    
def chatbot(state: State):
    return {"messages": [model.invoke(state.messages)]}

graph_builder = StateGraph(State)

#Adding Node
graph_builder.add_node("chat_bot", chatbot)

#Adding Edge
graph_builder.add_edge(START, "chat_bot")
graph_builder.add_edge("chat_bot", END)

#Compile
graph = graph_builder.compile()

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception as e:
#     print(f"Could not render graph: {e}")

response = graph.invoke(State(messages = [HumanMessage(content="Hi")]))

print(response["messages"])

for event in graph.stream(State(messages = [HumanMessage(content="Hi")])):
    for value in event["values"]:
        print(value["messages"][-1].content)