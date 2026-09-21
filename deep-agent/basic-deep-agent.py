from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient()
model = init_chat_model("anthropic:claude-sonnet-4.5")


def web_search(query: str, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = False):
    """Run a web search"""
    return tavily_client.search(query=query, max_results=max_results, include_raw_content=include_raw_content, topic=topic)


deep_agent = create_deep_agent(
    model = model,
    tools= [web_search],
    system_prompt="Act as a Researcher"
)

result = deep_agent.invoke({"messages": [{"role": "user", "content": "What is langgraph?"}]})
print(result["messages"][-1].content)