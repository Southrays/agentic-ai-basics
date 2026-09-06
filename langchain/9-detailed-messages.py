from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model('gpt-4o')

system_msg = SystemMessage("You are a senior python developer with expertise in web frameworks. Always provide code examples and explain your reasoning. Be concise but thorough in your explanations")

human_msg = HumanMessage(
    content="Hello",
    name="alice",
    id="msg_123"
)

messages = [
    system_msg,
    human_msg
]

response = model.invoke(messages)

print(response.content)
