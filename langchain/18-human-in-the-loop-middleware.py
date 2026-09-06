from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by it's id"""
    return f"Email content for {email_id}"

def send_email_tool(recepient: str, subject: str, body: str) -> str:
    """Mock function to send an Email"""
    return f"Email sent to {recepient} with {subject} and {body}"

agent = create_agent(
    model="anthropic:claude-sonnet-4.0",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions" : ["approve", "edit", "reject"]
                },

                "read_email_tool" : False
            }
        )
    ]
)

#For Testing Approve
config_approve: RunnableConfig = {"configurable": {"thread_id": "test-approve"}}

result_approve = agent.invoke({
    "messages": [HumanMessage(content="send email to johndoe@example.com with subject 'Hello' and body 'How are you?'")]
}, config_approve)


if "__interrupt__" in result_approve:
    print("Paused! Approving...")

    result_approve = agent.invoke(
        Command(
            resume = {
                "decisions": [
                    {"type": "approve"},
                ]
            }
        ), config_approve
    )

    print(f"Result: {result_approve["messages"][-1].content}")


#For Testing Reject
config_reject: RunnableConfig = {"configurable": {"thread_id": "test-reject"}}

result_reject = agent.invoke({
    "messages": [HumanMessage(content="send email to johndoe@example.com with subject 'Hello' and body 'How are you?'")]
}, config_reject)

if "__interrupt__" in result_reject:
    print("Paused! Approving...")

    result = agent.invoke(
        Command(
            resume= {
                "decisons": [
                    {"type": "reject"}
                ]
            }
        ), config_reject
    )

    print(f"Result: {result["messages"][-1].content}")


#For Testing Edit and Approve
config_edit: RunnableConfig = {"configurable": {"thread_id": "test-edit"}}

result_edit = agent.invoke({
    "messages": [HumanMessage(content="send email to wrongmail@example.com with subject 'Hello' and body 'How are you?'")]
}, config_edit)

if "__interrupt__" in result_edit:
    print("Paused! Approving...")

    result = agent.invoke(
        Command(
            resume= {
                "decisons": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "send_email_tool",
                            "args": {
                                "recepient": "correct@example.com",
                                "subject": "Corrected Subject",
                                "body": "This body was corrected"
                            }
                        }
                    }
                ]
            }
        ), config_edit
    )

    print(f"Result: {result["messages"][-1].content}")