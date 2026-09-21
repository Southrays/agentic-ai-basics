from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

model = init_chat_model("anthropic:claude-sonnet-4.5")

#Define a simple dummy tool
@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Search results for: {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send email to a recepient"""
    return f"Email sent to: {to}, with subject {subject}"

@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from the database"""
    return f"Deleted record from {table} where {condition}"

#Create agent with HITL Middleware
agent = create_agent(
    model=model,
    tools=[search_web, send_email, delete_records],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "search_web": False,
                "send_email": True,
                "delete records": True,
            }
        )
    ]
)

print("Human in the loop agent created")

config: RunnableConfig = {"configurable": {"thread_id": "session_001"}}

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "send an email to team@company.com about Q4 results"
    }]
}, config)


print("Agent Paused -- Awaiting human approval --")

approved_result = agent.invoke(
    Command(resume = {"decisions": [{"type": "approve"}]}),
    config=config
)

print("Approved -- Final response --")
print(approved_result["messages"][-1].content)



#Alternative - Human rejects
config2: RunnableConfig = {"configurable": {"thread_id": "session_002"}}

result2 = agent.invoke({
    "messages": [{"role": "user","content": "Delete all records from user table where active=false"}]
}, config=config2)

print("Agent Paused -- Awaiting human approval --")

rejected_result = agent.invoke(
    Command(resume = {"decisions": [{"type": "reject", "reason": "Too risky, needs DBA review"}]}),
    config=config2
)

print("Rejected -- Final response --")
print(rejected_result["messages"][-1].content)


