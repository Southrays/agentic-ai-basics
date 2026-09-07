import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

model = init_chat_model("anthropic:claude-sonnet-4.5")

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python", 
                "args": ["mathserver.py"], 
                "transport": "stdio"
            },

            "weather": {
                "url": "http://127.0.0.1:8000",
                "transport": "streamable_http"
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent(model, tools)

    math_response = agent.invoke({
        "messages": [{"role": "user", "content": "What is (3 + 5) x 12?"}]
    })

    print("Math response: ", math_response["messages"][-1].content)

    weather_response = agent.invoke({
        "messages": [{"role": "user", "content": "What is the weather condition in Tokyo?"}]
    })

    print("Weather response: ", weather_response["messages"][-1].content)

asyncio.run(main())