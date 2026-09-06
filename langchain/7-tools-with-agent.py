from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

@tool("get_weather", description="This takes in a location an returns the weather condition.")
def get_weather(city: str) -> str:
    return f"Its is sunny in {city}."

model = init_chat_model('gpt-4o')

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant"
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather in Cape Town?"}
    ]
})

print(response["messages"][-1].content)
