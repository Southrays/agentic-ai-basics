import os

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

document=Document(
    page_content="This is the main text content i am using to create RAG",
    metadata={
        "source": "example.py",
        "pages": 1,
        "author": "Southrays",
        "date_created": "21-03-2026"
    }
)

os.makedirs("../data/text-files", exist_ok=True)

sample_text={
    "../data/text-files/python-intro.txt":"""Python Programming Introduction

Python is a high-level, interpreted programming language known for it's simplicity and readability.
Created by Guido Van Rossum and first released in 1991, python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in web development, data science, artificial intelligence and automation. """,

    "../data/text-files/machine-learning.txt": """Machine learning basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve
from experiences without being explicitly programmed. It focuses on developing computer programs that
can access data and use it to learn for themselves.

Types of Machine learning:
1. Supervised learning: Learning with labeled data
2. Unsupervised learning: Finding patterns in unlabeled data
3. Reinforcement learning: Learning through rewards and penalties

    """
}

for filepath, content in sample_text.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("Sample files created.")

loader = TextLoader("../data/text-files/python-intro.txt", encoding="utf-8")
document = loader.load()