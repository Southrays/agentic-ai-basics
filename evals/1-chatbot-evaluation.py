import openai
from dotenv import load_dotenv
from langsmith import Client, wrappers

load_dotenv()
client = Client()
openai_client = wrappers.wrap_openai(openai.OpenAI())

# Define dataset: these are your test cases
dataset_name = "Chatbots Evaluation"
dataset = client.create_dataset(dataset_name)

examples = [
    {
        "question": "What is LangChain?",
        "answer": "A framework for building LLM applications",
    },
    {
        "question": "What is LangSmith?",
        "answer": "A platform for observing and evaluating LLM applications",
    },
    {
        "question": "What is OpenAI?",
        "answer": "A company that creates Large Language Models",
    },
    {
        "question": "What is Google?",
        "answer": "A technology company known for search",
    },
    {
        "question": "What is Mistral?",
        "answer": "A company that creates Large Language Models",
    },
]

for example in examples:
    client.create_example(
        inputs={"question": example["question"]},
        outputs={"answer": example["answer"]},
        dataset_id=dataset.id,
    )

eval_instructions = "You are an expert professor specialized in grading student's answers to questions."

def correctness(inputs, outputs, reference_outputs) -> dict:
    user_content = f"""You are grading the following question:
    {inputs['question']}
    Here is the real answer:
    {reference_outputs['answer']}
    You are grading the following predicted answer:
    {outputs['response']}
    Respond with CORRECT or INCORRECT:
    Grade:
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": eval_instructions},
            {"role": "user", "content": user_content},
        ],
    ).choices[0].message.content

    if response is None:
        return {"score": False}

    return {"score": response.strip().upper() == "CORRECT"}


def concision(outputs, reference_outputs) -> dict:
    return {
        "score": len(outputs["response"])
        < 2 * len(reference_outputs["answer"])
    }

default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."
def my_app(question: str, model: str = "gpt-4o-mini", instructions: str = default_instructions) -> str:
    response = openai_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    )

    if response.choices[0].message.content is not None:
        return response.choices[0].message.content

    return ""

### Call my_app for every datapoints
def ls_target(inputs: dict) -> dict:
    return {"response": my_app(inputs["question"])}

## Run our evaluation
experiment_results=client.evaluate(
    ls_target,
    data=dataset_name,
    evaluators=[correctness,concision],
    experiment_prefix="openai-4o-mini-chatbot"
)