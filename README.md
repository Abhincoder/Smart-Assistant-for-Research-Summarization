# Smart-Assistant-for-Research-Summarization
# Document-Aware AI Assistant

This Streamlit application allows you to upload PDF or TXT documents, get a summary, ask questions about its content, and even challenge yourself with automatically generated questions.

## Features

* **Document Upload:** Supports PDF and TXT file uploads [cite: uploaded:doc_sum.py].
* **Document Processing:** Chunks the uploaded document content and creates a vector store for efficient retrieval [cite: uploaded:doc_sum.py].
* **Document Summarization:** Generates a concise summary of the uploaded document [cite: uploaded:doc_sum.py].
* **Question & Answer:** Allows users to ask questions about the document, with answers derived directly from the document's content [cite: uploaded:doc_sum.py].
* **Challenge Mode:** Generates comprehension-based questions from the document and evaluates user answers [cite: uploaded:doc_sum.py].

## Setup Instructions

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

* **Python 3.8+** (or newer recommended)
* **pip** (Python package installer)

### 2. Create a Virtual Environment (Recommended)

It's good practice to use a virtual environment to manage dependencies for your project.

```bash
python -m venv .venv
pip install streamlit python-dotenv langchain-google-genai langchain-community chromadb pypdf tiktoken
# If you are using LangSmith for tracing (present in your imports, but optional for basic functionality):
# pip install langsmith


Architecture / Reasoning Flow
The application implements a Retrieval-Augmented Generation (RAG) pattern to provide answers and insights based on the content of uploaded documents.

Core Components:
Streamlit User Interface (UI): [cite: uploaded:doc_sum.py]
Manages user interaction, including file uploads (st.file_uploader), displaying document summaries and answers (st.info, st.markdown), and accepting user questions (st.text_input) [cite: uploaded:doc_sum.py].
Handles interactive elements like mode selection (st.radio) and buttons (st.button) [cite: uploaded:doc_sum.py].
Document Loading and Processing (load_and_process_document_in_memory): [cite: uploaded:doc_sum.py]
When a user uploads a PDF or TXT file, this function is called [cite: uploaded:doc_sum.py].
For PDF files, it temporarily saves the file to a unique temporary directory, uses PyPDFLoader to extract text, and then deletes the temporary files [cite: uploaded:doc_sum.py].
For TXT files, it reads the content directly into memory and wraps it as a LangChain Document [cite: uploaded:doc_sum.py].
RecursiveCharacterTextSplitter then breaks down the extracted text into smaller, overlapping chunks (e.g., 1000 characters with 200 character overlap) [cite: uploaded:doc_sum.py]. This helps manage context window limits of the LLM and improves retrieval accuracy.
Embedding Generation (GoogleGenerativeAIEmbeddings): [cite: uploaded:doc_sum.py]
Each text chunk generated is transformed into a high-dimensional numerical vector (an "embedding") using Google's embedding-001 model [cite: uploaded:doc_sum.py]. These embeddings capture the semantic meaning of the text.
Vector Store (Chroma): [cite: uploaded:doc_sum.py]
The generated embeddings and their corresponding text chunks are stored in a Chroma vector database [cite: uploaded:doc_sum.py]. Chroma indexes these embeddings, enabling quick and efficient similarity searches.
Large Language Model (LLM) (ChatGoogleGenerativeAI): [cite: uploaded:doc_sum.py]
The application uses a Google Gemini model (e.g., gemini-2.5-flash or gemini-pro) for all text generation tasks, including summarization, answering questions, generating challenge questions, and evaluating answers [cite: uploaded:doc_sum.py].
Reasoning Flow for Different Features:
A. Document Summarization (generate_summary) [cite: uploaded:doc_sum.py]
All processed document chunks are combined into a single text string (with a length limit of 5000 characters to fit the LLM's context window). [cite: uploaded:doc_sum.py]
A specific prompt instructing the LLM to summarize this text (e.g., "in no more than 150 words, focusing on key themes and information") is sent to the LLM. [cite: uploaded:doc_sum.py]
The LLM generates and returns the summary based on the provided text and prompt. [cite: uploaded:doc_sum.py]
B. Question & Answer (ask_question) [cite: uploaded:doc_sum.py]
Retrieval: When a user asks a question, the question is embedded and used to search the Chroma vector store (vector_store.as_retriever().invoke(query)). [cite: uploaded:doc_sum.py] This step retrieves the document chunks most semantically similar to the user's question. [cite: uploaded:doc_sum.py]
Contextual Prompting: The retrieved relevant document chunks are combined to form a Document Context. [cite: uploaded:doc_sum.py] A prompt is then constructed, providing this Document Context and the user's Question to the LLM. [cite: uploaded:doc_sum.py] The prompt instructs the LLM to answer only based on the provided context and to avoid using outside knowledge. [cite: uploaded:doc_sum.py]
Answer Generation: The LLM processes this contextualized prompt and generates the answer. [cite: uploaded:doc_sum.py]
C. Challenge Mode (generate_challenge_questions, evaluate_answer) [cite: uploaded:doc_sum.py]
Question Generation: [cite: uploaded:doc_sum.py]
A portion of the full document text (up to 5000 characters) is sent to the LLM. [cite: uploaded:doc_sum.py]
A prompt asks the LLM to generate three distinct, challenging, logic-based or comprehension-focused questions that can be answered solely from the document's content, formatted as a numbered list. [cite: uploaded:doc_sum.py]
The LLM's response is then parsed into individual questions, which are displayed to the user. [cite: uploaded:doc_sum.py]
Answer Evaluation: [cite: uploaded:doc_sum.py]
When the user provides an answer to a generated question: [cite: uploaded:doc_sum.py]
The original question is used to retrieve relevant chunks from the Chroma vector store. [cite: uploaded:doc_sum.py]
An evaluation_prompt is crafted, supplying the original question, the User's Answer, and the Document Context (retrieved chunks) to the LLM. [cite: uploaded:doc_sum.py]
The LLM is instructed to evaluate the user's answer for correctness only based on the provided context, providing feedback and justification. [cite: uploaded:doc_sum.py]




How to RUN
streamlit run doc_sum.py
