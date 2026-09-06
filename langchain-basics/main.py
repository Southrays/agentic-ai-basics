from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool("get_weather", description="Return weather information of a given city", return_direct=False)
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()


@tool("locate_user", description="Look up a user's city based on their context.")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case "ABC123":
            return "Port Harcourt"
        case "XYZ456":
            return "Lagos"
        case "LLL232":
            return "Abuja"
        case _:
            return "Unknown"
        

model = init_chat_model(
    model = "gpt-4o",
    temperature = 0.1
)

checkpointer = InMemorySaver()


agent = create_agent(
    model = model,
    tools = [get_weather, locate_user],
    system_prompt= "You are a weather assistant, who always cracks jokes and is humorous while remaining helpful.",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config: RunnableConfig = {
    "configurable": {
        "thread_id": "1"
    }
}

response = agent.invoke({
    "messages" : [
        { "role": "user", "content": "What is the weather like?"}
    ]
},
    config= config,
    context= Context(user_id="ABC123")
)

print(response["structured_response"])