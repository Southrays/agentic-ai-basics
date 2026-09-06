from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model= "gpt-4o",
    temperature= 0.7
)

conversation = [
    SystemMessage("You are a helpful assistant for questions regarding programming"),
    HumanMessage("What is Python?"),
    AIMessage("Python is an interpreted programming language")
]

response = model.invoke(conversation)