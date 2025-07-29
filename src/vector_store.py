from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import os

def build_vectorstore(texts, metadatas=None, embedding_model=None):
    """
    Build a FAISS vector store from raw texts and optional metadata.
    """
    documents = [
        Document(page_content=text, metadata=metadatas[i] if metadatas else {})
        for i, text in enumerate(texts)
    ]
    return FAISS.from_documents(documents, embedding_model)

def save_vectorstore(vectorstore, path="faiss_store"):
    """
    Save the FAISS vector store to local disk.
    """
    vectorstore.save_local(path)

def load_vectorstore(path="faiss_store", embedding_model=None):
    """
    Load the FAISS vector store from local disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No FAISS store found at: {path}")
    return FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True # use for local testing only
    )

