
# Splits raw PDF content into overlapping text chunks

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path, chunk_size=500, chunk_overlap=50):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    return chunks
