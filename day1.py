from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

docs = PyPDFLoader("data/handbook_30pages.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(len(chunks))
print(chunks[0])
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
QDRANT_URL = os.getenv("QDRANT_URL")
store = QdrantVectorStore.from_documents(
    documents=chunks,
    collection_name="handbook",
    embedding=embedding_model,
    url=QDRANT_URL,
)
print("store : ",store)
