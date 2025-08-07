from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

def build_qa_chain(vectorstore: FAISS) -> RetrievalQA:

    pipe = pipeline(
        task="text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=823,
        temperature=0.1,
        do_sample=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # can change to "map_reduce" or "refine" for different behaviors
        retriever=retriever,
        return_source_documents=True
    )

    return chain
