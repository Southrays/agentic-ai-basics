from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

def get_weather(city: str) -> str:
    '''Return weather information of a given city'''
    return f'The weather in {city} is sunny.'

agent = create_agent(
    model='gpt-4o',
    tools=[get_weather],
    system_prompt="You are a helpful assistant"
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather like in New York"}
    ]
})

print(response["messages"][-1].content)