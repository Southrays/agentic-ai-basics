import os
import subprocess
import uuid
from typing import Annotated, Literal, NotRequired, TypedDict, cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model(model= "gpt-4o")

KNOWLEDGE = [
    "Neuraline is a youtube channel focused on programming, AI and software engineering tutorials.",
    "Langgraph is a library for building stateful, multi-agent applications on top of Langchain.",
    "A stategraph in Langgraph defines nodes and edges that operate on a shared typed state.",
    "Checkpointers like InMemorySaver let Langgraph persist conversation state accross invocations using thread_id.",
    "RAG (Retrieval-Augumented Generation) combines a retriever over a knowledge base with an LLM to ground answers in source documents."
]

vector_store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])

class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(..., description= "Classify whether the user wants to just chat, ask for knowledge or change code in the project.")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_intent: NotRequired[str | None]

    next_node: NotRequired[str | None]

def classify_intent(state: State):
    structured_output = model.with_structured_output(IntentClassifier)
    result = cast(IntentClassifier, structured_output.invoke([
        {"role": "system", "content": "Determine/Classify whether the user wants to chat ('chat'), retrieve a knowlwdge ('knowledge') or change code ('code')." },
        {"role": "user", "content": state["messages"][-1].content}
    ]))

    return {"message_intent": result.message_intent}

def accept_coding(state: State):
    user_prompt = str(state["messages"][-1].content)
    decision = interrupt(f'About to run Claude Code with request:\n\n{user_prompt}\n\nApprove? (Yes/No or type a revised request)')

    text = str(decision).strip().lower()

    if text in ["y", "yes", "approve", "ok", "continue", "proceed"]:
        return {"next_node": "code_agent"}
    
    if text in ["n", "no", "cancel", "deny", "stop", "reject"]:
        return {"messages": [{"role": "assistant", "content": "Coding request was denied by the user."}], "next_node": "denied"}

    return {"messages": [{"role": "user", "content": text}], "next_node": "accept_coding"}

def prepare_coding_request(state: State):
    messages = [
        {"role": "system", "content": "Rewrite the latest user coding request into a clear instruction for Claude Code. Use the conversation history as a context. Only output the instruction, no explanation"}
    ] + state["messages"]

    response = model.invoke(messages)

    return {"messages": [{"role": "user", "content": response.content}]}


def prompt_llm_chat(state: State):
    messages = [{"role": "system", "content": "You are a talkative chatbot for fun. Be nice"}] + state["messages"]
    response = model.invoke(messages)

    return {"messages": [{"role": "assistant", "content": response.content}]}

def prompt_llm_rag(state: State):
    query = str(state['messages'][-1].content)
    documents = vector_store.similarity_search(query, k=3)

    context = '\n'.join(f'- {doc.page_content}' for doc in documents)

    messages = [
        {'role': 'system', 'content': f"You are a RAG agent. Answer the user using only the context below. If the answer is not in it, say you don't know.\n\nContext:{context}"}
    ] + state['messages']
    response = model.invoke(messages)

    return {"messages": [{"role": "assistant", "content": response.content}]}

def prompt_llm_code(state: State):
    user_prompt = str(state["messages"][-1].content)
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")

    result = subprocess.run(
        ["claude", "-p", user_prompt, "--permission-mode", "acceptEdits"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False
    )

    output = result.stdout.strip() or result.stderr.strip()

    return {"messages": [{"role": "assistant", "content": output}]}


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("code_agent", prompt_llm_code)
graph_builder.add_node("prepare_coding_request", prepare_coding_request)
graph_builder.add_node("accept_coding", accept_coding)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("prepare_coding_request", "accept_coding")

graph_builder.add_conditional_edges("accept_coding", lambda state: "end" if state.get("next_node") == "denied" else state["next_node"], {"end": END, "code": "code_agent", "accept_coding": "prepare_coding_request"})
graph_builder.add_conditional_edges("classifier", lambda state: state["message_intent"], {"chat": "chat_agent", "knowledge": "rag_agent", "code": "prepare_coding_request"})

graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("code_agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config: RunnableConfig = {
    "configurable": {
        "thread_id": str(uuid.uuid4())
    }
}

while True:
    user_message = input("Enter message: ")
    result = graph.invoke({ "messages": [HumanMessage(content=user_message)]}, config=config)

    while "__interrupt__" in result:
        prompt = result["__interrupt__"][0].value
        decision = input(f'{prompt}\n> ')
        result = graph.invoke(Command(resume=decision))

    print(result["messages"][-1].content)