
""""
Splits raw PDF content into overlapping text chunks which is useful both for indexing data and passing it into a model
"""
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path, chunk_size=500, chunk_overlap=50):
    #load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    """
    Split the PDF into chunks
    some different split methoeds:
        1. Length-based
            - Token-based: CharacterTextSplitter
            - Character-based
        2. Text-structured based: RecursiveCharacterTextSplitter
        3. Document-structured based: MarkdownTextSplitter, CSVTextSplitter, ...
        4. Semantic meaning based: actually considers the content of the text
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    return chunks

"""
Chunk 1: “This is an introduction to neural networks...”
Chunk 2: “The architecture includes layers such as input, hidden...”
"""
