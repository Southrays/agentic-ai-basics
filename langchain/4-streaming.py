from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model('gpt-4o')

for chunk in model.stream("Write me a 200 word paragraph on Artificial Intelligence"):
    print(chunk.text)