import uuid

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

llm = init_chat_model(model="gpt-4o")

def prompt_llm(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, "prompt_llm")
graph_builder.add_edge("prompt_llm", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer= checkpointer)

config: RunnableConfig = {
    "configurable": {
        "thread_id": str(uuid.uuid4())
    }
}


while True:
    user_message = input("Enter message: ")
    print(graph.invoke({ "messages": [HumanMessage(content=user_message)]}, config=config))