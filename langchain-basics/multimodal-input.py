from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model = "gpt-4o",
    temperature = 0.5
)

message = {
    "role": "user", 
    "content": [
        { "type" : "text", "text": "Describe the contents of this image."},
        { 
            "type" : "image", 
            "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQftuzlSuZeG8AIohasnoSwYVNuHnmZ9QzIgNLtf3wdIIffT6fb7fJ2sUTq&s=10",
            "mime_type": "image/png"
        },
    ]
}