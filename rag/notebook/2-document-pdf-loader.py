from langchain_community.document_loaders import PyPDFLoader
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

pdf_loader = PyPDFLoader("../data/pdf-files/Ray_Elenwo_Smart_Contract_Engineer_Resume.pdf")

document = pdf_loader.load()

print(document)