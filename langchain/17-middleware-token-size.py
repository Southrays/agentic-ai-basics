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
            trigger=("tokens", 550),
            keep=("tokens", 200)
        )
    ],
    system_prompt="You are a helpful and fun mathematics assistant"
)

config: RunnableConfig = {"configurable": {"thread_id": "user-1"}}

def count_tokens(messages):
    total_chars = sum(len(str(m.content)) for m in messages)
    return total_chars // 4 

cities = [
    "Paris",
    "London",
    "New York",
    "Tokyo",
    "Dubai",
    "Singapore"
]

for city in cities:
    response = agent.invoke({
        "messages": [HumanMessage(content=f"Find hotels in {city}")]
    }, config)

    tokens = count_tokens(response["messages"])
    print(f"{city}: ~{tokens} tokens, {len(response["messages"])} messages")
    print(f"{(response['messages'])}")

