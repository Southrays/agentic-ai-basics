from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model('gpt-4o')

messages = [
    SystemMessage("You are a funny hip hop lyricist"),
    HumanMessage("Write an 8 bar on Artificial Intelligence")
]

response = model.invoke(messages)

print(response.content)
