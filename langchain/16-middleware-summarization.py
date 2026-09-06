from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

agent = create_agent(
    model="anthropic:claude-haiku-4-5-20251001",
    checkpointer = InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="anthropic:claude-haiku-4-5-20251001",
            trigger=("messages", 10),
            keep=("messages", 4)
        )
    ],
    system_prompt="You are a helpful and fun mathematics assistant"
)

config: RunnableConfig = {"configurable": {"thread_id": "user-1"}}

questions = [
    "What is 1 + 2?",
    "What is 2 + 2?",
    "What is 3 + 3?",
    "What is 4 + 4?",
    "What is 5 + 5?",
    "What is 6 + 6?",
]

for q in questions:
    response = agent.invoke({"messages": [HumanMessage(content=q)]}, config)
    print(f"Messages: {response}")
    print(f"Messages: {len(response['messages'])}")
