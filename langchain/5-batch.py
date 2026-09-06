from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model('gpt-4o')

responses = model.batch([
    "How are you?,",
    "Which continent is Japan in?",
    "What are the various classes of food?"
], config = {"max_concurrency": 5}) #This is optional: can be used to set a limit of how many messages per batch

for response in responses:
    print(response.content)