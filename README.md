# Parallel and Distributed RAG-Based Chatbot

This project is a simple Retrieval-Augmented Generation (RAG) chatbot for document question answering. It demonstrates concepts from Parallel and Distributed Computing by processing document chunks in parallel and searching relevant chunks efficiently.

## Features
- Loads text documents from the `data/` folder
- Splits documents into chunks using parallel processing
- Builds a lightweight TF-IDF style vector index without external AI APIs
- Retrieves the most relevant chunks for a user question
- Generates a grounded answer from retrieved text
- Supports command-line interaction

## Project Structure
```text
rag_pdc_project/
├── data/
│   └── course_notes.txt
├── output/
├── src/
│   ├── document_loader.py
│   ├── rag_engine.py
│   └── main.py
├── requirements.txt
└── README.md
```

## How to Run

1. Open terminal inside the project folder.
2. Run:
```bash
python src/main.py
```

3. Ask questions such as:
```text
What is parallel computing?
What is distributed computing?
How does RAG reduce hallucination?
Where is parallelism used in this project?
```

## Parallel Computing Concept Used
The project uses `ProcessPoolExecutor` to split and preprocess documents in parallel. This means different parts of the document can be processed at the same time instead of one by one.

## Distributed Computing Concept Used
The report explains how this prototype can be extended to a distributed system by using multiple worker machines, distributed vector databases, and concurrent request handling.
