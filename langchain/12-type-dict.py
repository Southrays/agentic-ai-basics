from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("gpt-4o")

class Actor(TypedDict):
    name: Annotated[str, ..., "Name of the actor"]
    role: Annotated[str, ..., "Role of the actor in the movie"]

class MovieDetails(TypedDict):
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The year the movie was released"]
    cast: Annotated[list[Actor], ..., "The actors of the movie"]
    genres: Annotated[list[str], ..., "The genres of the movie"]
    budget: Annotated[float, ..., "The budget of the movie in USD"]

model_with_typedict = model.with_structured_output(MovieDetails)

response = model_with_typedict.invoke("Provide details of the movie Inception")

print(response)