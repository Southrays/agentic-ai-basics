from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

chat_model = init_chat_model('gpt-4o')

response = chat_model.invoke("How are you?")

print(response.content)