from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

model = init_chat_model('gpt-4o')

@tool("get_weather", description="This takes in a location an returns the weather condition.")
def get_weather(city: str) -> str:
    return f"Its is sunny in {city}."


model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What is the weather in Cape Town?")

print(response)

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call["name"]}")
    print(f"Args: {tool_call["args"]}")
