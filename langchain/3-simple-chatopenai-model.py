from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

chat_model = ChatOpenAI(model='gpt-4o')

response = chat_model.invoke("How are you?")

print(response.content)