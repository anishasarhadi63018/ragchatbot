import os
import re
import math
import collections
import concurrent.futures
import streamlit as st
from openai import OpenAI

# ============================================================
# RAG-Based Chatbot for PDC CCP
# Topic: Parallel and Distributed Computing
# Streamlit + OpenAI LLM Version
# Main file name: pdc.py
# ============================================================

st.set_page_config(
    page_title="RAG Chatbot for PDC",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------
# Default Knowledge Base
# Used if no documents folder or .txt files exist.
# -----------------------------
DEFAULT_DOCUMENTS = [
    {
        "title": "Parallel Computing",
        "content": """
        Parallel computing is a computing technique in which a large problem is divided into smaller parts
        and these parts are processed at the same time using multiple processors or cores. The main goal of
        parallel computing is to increase speed, improve performance, and reduce execution time. Examples
        include multicore processors, GPU computing, scientific simulations, image processing, and big data
        processing.
        """
    },
    {
        "title": "Distributed Computing",
        "content": """
        Distributed computing is a model in which multiple computers connected through a network work together
        to solve a problem. Each computer is called a node. These nodes communicate by passing messages.
        Distributed systems are used in cloud computing, web applications, banking systems, Google services,
        online shopping platforms, and large-scale databases.
        """
    },
    {
        "title": "Difference Between Parallel and Distributed Computing",
        "content": """
        Parallel computing usually runs on multiple processors or cores inside one system, while distributed
        computing runs on multiple independent computers connected through a network. Parallel computing focuses
        mainly on speed and performance. Distributed computing focuses on resource sharing, scalability,
        availability, and fault tolerance.
        """
    },
    {
        "title": "RAG Chatbot",
        "content": """
        RAG stands for Retrieval-Augmented Generation. A RAG chatbot first retrieves relevant information from
        a knowledge base and then generates an answer using that information. In this project, the chatbot
        retrieves the most relevant document chunks based on the user's question and sends those chunks to an
        LLM to create a better answer.
        """
    },
    {
        "title": "Document Chunking",
        "content": """
        Document chunking means dividing large text documents into smaller parts called chunks. Chunking helps
        the chatbot search information more accurately. Instead of comparing the question with a full document,
        the chatbot compares the question with smaller chunks and retrieves the most relevant ones.
        """
    },
    {
        "title": "Parallel Document Processing",
        "content": """
        Parallel document processing means processing multiple documents or chunks at the same time. This can
        reduce processing time when the dataset is large. In Python, concurrent.futures can be used to process
        text chunks in parallel using ThreadPoolExecutor.
        """
    },
    {
        "title": "Vector Generation",
        "content": """
        Vector generation means converting text into numerical form so the computer can compare meanings or
        similarity. In this simple project, text is converted into word-frequency vectors. Each vector represents
        how often important words appear in a chunk.
        """
    },
    {
        "title": "Similarity-Based Retrieval",
        "content": """
        Similarity-based retrieval is the process of finding the most relevant text chunks for a user's question.
        Cosine similarity is commonly used for this purpose. It compares the question vector with document chunk
        vectors and returns the chunks with the highest similarity score.
        """
    },
    {
        "title": "Cosine Similarity",
        "content": """
        Cosine similarity measures how similar two vectors are. A higher cosine similarity means the question
        and document chunk are more related. It is widely used in search engines, recommendation systems,
        and chatbot retrieval systems.
        """
    },
    {
        "title": "Benefits of RAG",
        "content": """
        RAG improves chatbot answers by using external knowledge instead of relying only on fixed responses.
        It can reduce wrong answers, provide context-based responses, and work with custom documents such as
        course notes, PDFs, reports, and project files.
        """
    }
]


# -----------------------------
# Load Documents
# -----------------------------
def load_documents(folder_path="documents"):
    documents = []

    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".txt"):
                file_path = os.path.join(folder_path, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()

                    if content.strip():
                        documents.append({
                            "title": filename,
                            "content": content
                        })

                except Exception:
                    pass

    if not documents:
        documents = DEFAULT_DOCUMENTS

    return documents


# -----------------------------
# Chunk Documents
# -----------------------------
def chunk_text(text, chunk_size=90):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def process_single_document(document):
    clean_content = clean_text(document["content"])
    chunks = chunk_text(clean_content)

    processed_chunks = []

    for chunk in chunks:
        processed_chunks.append({
            "title": document["title"],
            "chunk": chunk
        })

    return processed_chunks


# -----------------------------
# Parallel Chunk Processing
# -----------------------------
@st.cache_data
def build_chunks():
    documents = load_documents()
    all_chunks = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_single_document, documents)

    for result in results:
        all_chunks.extend(result)

    return all_chunks


# -----------------------------
# Vector Generation
# -----------------------------
def text_to_vector(text):
    words = clean_text(text).split()
    return collections.Counter(words)


# -----------------------------
# Cosine Similarity
# -----------------------------
def cosine_similarity(vector1, vector2):
    common_words = set(vector1.keys()) & set(vector2.keys())

    numerator = sum(vector1[word] * vector2[word] for word in common_words)

    sum1 = sum(value ** 2 for value in vector1.values())
    sum2 = sum(value ** 2 for value in vector2.values())

    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if denominator == 0:
        return 0.0

    return numerator / denominator


# -----------------------------
# Retrieve Relevant Chunks
# -----------------------------
def retrieve_chunks(question, chunks, top_k=3):
    question_vector = text_to_vector(question)
    scored_chunks = []

    for item in chunks:
        chunk_vector = text_to_vector(item["chunk"])
        score = cosine_similarity(question_vector, chunk_vector)

        scored_chunks.append({
            "title": item["title"],
            "chunk": item["chunk"],
            "score": score
        })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]


# -----------------------------
# Generate Answer with OpenAI LLM
# -----------------------------
def generate_answer(question, retrieved_chunks):
    useful_chunks = [item for item in retrieved_chunks if item["score"] > 0]

    if not useful_chunks:
        return (
            "I could not find a strong match in the knowledge base. "
            "Please ask about parallel computing, distributed computing, RAG, "
            "chunking, vectors, or similarity-based retrieval."
        )

    context = "\n\n".join(
        f"Source: {item['title']}\nText: {item['chunk']}"
        for item in useful_chunks
    )

    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful academic chatbot for a Parallel and Distributed Computing project. "
                        "Answer in simple student-friendly language. Use only the provided context. "
                        "If the answer is not in the context, say that the knowledge base does not contain enough information."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question:\n{question}\n\n"
                        "Now give a clear answer."
                    )
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except KeyError:
        return (
            "OpenAI API key is missing. Please add it in Streamlit Secrets like this:\n\n"
            'OPENAI_API_KEY = "your_api_key_here"'
        )

    except Exception as e:
        return f"OpenAI error: {e}"


# -----------------------------
# Streamlit User Interface
# -----------------------------
st.title("🤖 RAG-Based Chatbot for PDC CCP")
st.write("### Topic: Parallel and Distributed Computing")
st.write(
    "This chatbot uses document chunking, vector generation, cosine similarity, "
    "retrieval-based search, and OpenAI LLM answer generation."
)

with st.sidebar:
    st.header("📌 Project Info")
    st.write("**Project:** RAG-Based Chatbot")
    st.write("**Subject:** Parallel and Distributed Computing")
    st.write("**LLM:** OpenAI")
    st.write("**Concepts Used:**")
    st.write("1. Parallel document processing")
    st.write("2. Document chunking")
    st.write("3. Vector generation")
    st.write("4. Similarity-based retrieval")
    st.write("5. OpenAI answer generation")

    st.header("📁 Optional Documents")
    st.write(
        "To use your own data, create a folder named `documents` in GitHub "
        "and add `.txt` files inside it."
    )

chunks = build_chunks()

col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_input(
        "Ask your question here:",
        placeholder="Example: What is parallel computing?"
    )

    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please type a question first.")
        else:
            retrieved = retrieve_chunks(question, chunks)
            answer = generate_answer(question, retrieved)

            st.subheader("✅ Chatbot Answer")
            st.write(answer)

            st.subheader("🔍 Retrieved Chunks")
            for index, item in enumerate(retrieved, start=1):
                with st.expander(
                    f"Chunk {index} | Source: {item['title']} | Score: {item['score']:.3f}"
                ):
                    st.write(item["chunk"])

with col2:
    st.subheader("📊 Knowledge Base")
    st.metric("Total Chunks", len(chunks))
    st.metric("Documents Used", len(set(item["title"] for item in chunks)))

    st.subheader("💡 Try Questions")
    st.write("- What is parallel computing?")
    st.write("- What is distributed computing?")
    st.write("- What is RAG?")
    st.write("- What is cosine similarity?")
    st.write("- What is document chunking?")
    st.write("- Difference between parallel and distributed computing?")

st.markdown("---")
st.write("Made with Streamlit + OpenAI for PDC CCP Project 🚀")
