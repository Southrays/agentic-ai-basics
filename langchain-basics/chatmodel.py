from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model= "gpt-4o",
    temperature= 0.7
)

response = model.invoke("Hello, what is python")

print(response)
print(response.content)