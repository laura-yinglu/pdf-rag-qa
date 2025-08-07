from langchain.embeddings import HuggingFaceEmbeddings

# Returns an instance of the OpenAI embedding model.
# This will convert input text into dense vector representations.

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )
""""
(1) Embed text as a vector: Embeddings transform text into a numerical vector representation.

(2) Measure similarity: Embedding vectors can be compared using simple mathematical operations.
Some common similarity metrics include:

Cosine Similarity: Measures the cosine of the angle between two vectors.
Euclidean Distance: Measures the straight-line distance between two points.
Dot Product: Measures the projection of one vector onto another.


Chunk 1 → [0.12, -0.43, ..., 0.99]   ← vector of 1536 dims (OpenAI)
Chunk 2 → [0.55, -0.01, ..., -0.28]

our question will also be embedded into a vector:
vector = get_embedding_model().embed_documents([chunk1， chunk2])[0] // output first one
Query: “What are neural network layers?” → [0.52, -0.31, ..., 0.03]

"""