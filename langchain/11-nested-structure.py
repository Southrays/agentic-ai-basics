from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("gpt-4o")

class Actor(BaseModel):
    name: str = Field(description="Name of the actor")
    role: str = Field(description="Role of the actor in the movie")

class MovieDetails(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    cast: list[Actor] = Field(description="The actors of the movie")
    genres: list[str] = Field(description="The genres of the movie")
    budget: float = Field(description="The budget of the movie in USD")

model_with_structure = model.with_structured_output(MovieDetails)

response = model_with_structure.invoke("Provide details of the movie Inception")

print(response)