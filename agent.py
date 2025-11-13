# agent.py
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from config import INDEX_A, INDEX_B, SENTENCE_MODEL, OLLAMA_MODEL
import time

# load embeddings & retrievers
embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{SENTENCE_MODEL}")
db_a = FAISS.load_local(str(INDEX_A), embeddings, allow_dangerous_deserialization=True)
db_b = FAISS.load_local(str(INDEX_B), embeddings, allow_dangerous_deserialization=True)

retriever_a = db_a.as_retriever(search_kwargs={"k": 4})
retriever_b = db_b.as_retriever(search_kwargs={"k": 4})

# Ollama model (synchronous)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

def format_docs(docs):
    formatted = []
    for d in docs:
        company = d.metadata.get("company", "Unknown Company")
        formatted.append(f"[{company}]\n{d.page_content}")
    return "\n\n".join(formatted)


def llm_route_question(question: str) -> str:
    """
    Ask the LLM to return A, B, or BOTH.
    """
    prompt = f"""
You are a router. Given the question, return the word A if it concerns Company A,
B if it concerns Company B, or BOTH if it requires both companies.

Answer with only one token: A, B, or BOTH.

Question: {question}
"""
    resp = llm.invoke(prompt)
    text = resp.content.strip().upper()
    if "BOTH" in text:
        return "BOTH"
    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    # fallback: BOTH
    return "BOTH"

def retrieve_context_for_route(route: str, question: str) -> str:
    docs = []
    if route in ("A", "BOTH"):
        docs += retriever_a.invoke(question)
    if route in ("B", "BOTH"):
        docs += retriever_b.invoke(question)
    return format_docs(docs)


def generate_answer_stream(question: str, context: str):
    """
    Try to stream tokens from the LLM if available, otherwise return final text.
    Yields token strings.
    """
    prompt = f"""
            You are a financial analyst answering questions about Company A and Company B.

            Each section of context begins with a tag like [CompanyA] or [CompanyB].
            Use the information from the correct company to answer accurately.

            If the question compares both, use both sets of data.

            Context:
            {context}

            Question: {question}

            Answer in a concise factual sentence, using numeric values from context when available.
            If context is unclear, make the best inference from the data instead of refusing to answer.
            """


    # If model supports a .stream method, use it (langchain_ollama may provide model.stream)
    if hasattr(llm, "stream"):
        # model.stream yields chunks — adapt to your installed version
        for chunk in llm.stream(prompt):
            # each chunk is a ChatGenerationChunk or similar
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
            elif hasattr(chunk, "text") and chunk.text:
                yield chunk.text
            elif isinstance(chunk, str):
                yield chunk

        return

    # Fallback synchronous invocation
    resp = llm.invoke(prompt)
    yield resp.content

# Convenience combined call
def answer_question(question: str):
    route = llm_route_question(question)
    context = retrieve_context_for_route(route, question)
    return route, context, generate_answer_stream(question, context)
