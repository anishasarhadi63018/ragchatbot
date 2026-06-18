import os
import math
import re
import collections
import concurrent.futures


# -------------------------------
# RAG-Based Chatbot for PDC CCP
# -------------------------------
# Topic: Parallel and Distributed Computing
# Concept Used:
# 1. Parallel document chunk processing
# 2. Parallel embedding/vector generation
# 3. Similarity-based retrieval
# 4. Question answering using retrieved chunks


# ---------- Text Cleaning ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


# ---------- Load Documents ----------
def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append({
                    "filename": filename,
                    "content": content
                })

    return documents


# ---------- Split Text into Chunks ----------
def split_into_chunks(text, chunk_size=80):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ---------- Process One Document ----------
def process_document(document):
    chunks = split_into_chunks(document["content"])

    processed_chunks = []

    for index, chunk in enumerate(chunks):
        processed_chunks.append({
            "filename": document["filename"],
            "chunk_id": index,
            "text": chunk
        })

    return processed_chunks


# ---------- Parallel Chunk Processing ----------
def parallel_chunk_processing(documents):
    all_chunks = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_document, documents)

    for doc_chunks in results:
        all_chunks.extend(doc_chunks)

    return all_chunks


# ---------- Create Vector ----------
def create_vector(text):
    cleaned = clean_text(text)
    words = cleaned.split()
    return collections.Counter(words)


# ---------- Parallel Vector Generation ----------
def parallel_vector_generation(chunks):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        vectors = list(executor.map(lambda chunk: create_vector(chunk["text"]), chunks))

    for i in range(len(chunks)):
        chunks[i]["vector"] = vectors[i]

    return chunks


# ---------- Cosine Similarity ----------
def cosine_similarity(vector1, vector2):
    common_words = set(vector1.keys()) & set(vector2.keys())

    numerator = sum(vector1[word] * vector2[word] for word in common_words)

    sum1 = sum(value ** 2 for value in vector1.values())
    sum2 = sum(value ** 2 for value in vector2.values())

    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if denominator == 0:
        return 0

    return numerator / denominator


# ---------- Retrieve Relevant Chunks ----------
def retrieve_chunks(question, chunks, top_k=3):
    question_vector = create_vector(question)

    scored_chunks = []

    for chunk in chunks:
        score = cosine_similarity(question_vector, chunk["vector"])
        scored_chunks.append({
            "filename": chunk["filename"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": score
        })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks[:top_k]


# ---------- Generate Answer ----------
def generate_answer(question, retrieved_chunks):
    if retrieved_chunks[0]["score"] == 0:
        return "Sorry, I could not find relevant information in the documents."

    answer = "Based on the retrieved document content:\n\n"

    for i, chunk in enumerate(retrieved_chunks, start=1):
        answer += f"Source {i}: {chunk['filename']} | Chunk {chunk['chunk_id']}\n"
        answer += f"{chunk['text']}\n\n"

    return answer


# ---------- Main Chatbot ----------mkdir dat
def main():
    print("=" * 60)
    print("RAG-Based Chatbot for Parallel and Distributed Computing")
    print("=" * 60)

    folder_path = "data"

    if not os.path.exists(folder_path):
        print("Data folder not found. Please create a folder named 'data'.")
        return

    documents = load_documents(folder_path)

    if not documents:
        print("No .txt files found in the data folder.")
        return

    print("\nLoading documents...")
    print(f"Total Documents Loaded: {len(documents)}")

    print("\nProcessing documents in parallel...")
    chunks = parallel_chunk_processing(documents)
    print(f"Total Chunks Created: {len(chunks)}")

    print("\nGenerating vectors in parallel...")
    chunks = parallel_vector_generation(chunks)

    print("\nChatbot is ready!")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower() == "exit":
            print("Chatbot closed.")
            break

        retrieved_chunks = retrieve_chunks(question, chunks)
        answer = generate_answer(question, retrieved_chunks)

        print("\n" + answer)
        print("-" * 60)


if __name__ == "__main__":
    main()