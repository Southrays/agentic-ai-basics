from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

class ContactInfo(BaseModel):
    name: str = Field(description="Name of the person")
    email: str = Field(description="Email of the person")
    phone: str = Field(description="Phone number of the person")

model = init_chat_model("anthropic:claude-haiku-4-5-20251001")

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a helpful assistant",
    response_format=ContactInfo
)

result = agent.invoke({
    "messages": [{ "role": "user", "content": "Extract the contact information from John Doe, john@example.com, (+234)813372"}]
})

print(result["structured_response"])