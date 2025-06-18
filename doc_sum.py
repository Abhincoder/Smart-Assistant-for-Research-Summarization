from asyncio.log import logger
import asyncio
import os
import json
import uuid
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone
import streamlit as st
from langsmith import Client
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader 
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from io import BytesIO 

import tempfile
import shutil # For rmtree
# Load environment variables
load_dotenv()

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "google_api_key_set" not in st.session_state:
    st.session_state.google_api_key_set = False

# Retrieve and check API key at the beginning
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    st.error("GOOGLE_API_KEY environment variable not found. Please set it in your .env file or system environment.")
    st.stop()
else:
    st.session_state.google_api_key_set = True


def load_and_process_document_in_memory(uploaded_file_object):
    file_type = uploaded_file_object.name.split('.')[-1].lower()
    
    
    if file_type == "pdf":
        
        temp_dir_for_pdf = tempfile.mkdtemp()
        temp_pdf_path = os.path.join(temp_dir_for_pdf, uploaded_file_object.name)
        
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file_object.getbuffer()) # Write content to temp file
        
        try:
            loader = PyPDFLoader(temp_pdf_path)
            pages = loader.load()
            chunks = text_splitter.split_documents(pages)
            return chunks
        finally:

            shutil.rmtree(temp_dir_for_pdf)

    elif file_type == "txt":
      
        raw_text = uploaded_file_object.read().decode("utf-8")
        
        from langchain.docstore.document import Document
        doc = Document(page_content=raw_text, metadata={"source": uploaded_file_object.name})
        chunks = text_splitter.split_documents([doc])
        return chunks
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        return None

def generate_summary(chunks):
    full_text = " ".join([chunk.page_content for chunk in chunks])
    summary_prompt = f"Summarize the following text in no more than 150 words, focusing on key themes and information:\n\n{full_text[:5000]}"
    summary_response = llm.invoke(summary_prompt)
    return summary_response.content


def ask_question(query, vector_store):
    if not vector_store:
        return "Please upload a document first."

    retriever = vector_store.as_retriever()
    relevant_docs = retriever.invoke(query) 
    context = "\n".join([doc.page_content for doc in relevant_docs])

    qa_prompt = f"""Based on the following document context, answer the question.
    Ensure your answer is directly supported by the provided text.
    Do not use outside knowledge.

    Document Context:
    {context}

    Question: {query}

    Answer:"""

    response = llm.invoke(qa_prompt)
    return response.content


def generate_challenge_questions(chunks):
    full_text = " ".join([chunk.page_content for chunk in chunks])
    q_gen_prompt = f"""Given the following document, generate three distinct, challenging
    logic-based or comprehension-focused questions that can be answered solely from its content.
    Format them as a numbered list.

    Document:
    {full_text[:5000]}

    Questions:"""
    response = llm.invoke(q_gen_prompt)

    questions = [q.strip() for q in response.content.split('\n') if q.strip()]
    return questions if questions else ["Could not generate questions. The document might be too short or lack sufficient content."]



def evaluate_answer(question, user_answer, vector_store):
    if not vector_store:
        return "Error: Document not loaded."

    retriever = vector_store.as_retriever()
    relevant_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    evaluation_prompt = f"""Question: {question}
    User's Answer: {user_answer}
    Document Context: {context}

    Based *only* on the provided Document Context, evaluate the User's Answer for correctness.
    Provide feedback and justify your evaluation by referencing the document (e.g., "This is supported by Section 2, page 5.").
    State clearly if the answer is correct or incorrect.

    Evaluation:"""

    response = llm.invoke(evaluation_prompt)
    return response.content

# --- Streamlit UI ---
st.set_page_config(page_title="Document-Aware AI Assistant")
st.title("📄 Document-Aware AI Assistant")

if st.session_state.google_api_key_set: # Only show UI if API key is set

  
    uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

    if uploaded_file is not None:
        st.write(f"Processing '{uploaded_file.name}'...")
        with st.spinner("Loading and processing document..."):
            # Pass the uploaded_file object directly to the modified function
            st.session_state.document_chunks = load_and_process_document_in_memory(uploaded_file)
            if st.session_state.document_chunks:
                st.session_state.vector_store = Chroma.from_documents(
                    documents=st.session_state.document_chunks,
                    embedding=embeddings
                )
                st.success("Document processed and ready!")
                st.session_state.summary = generate_summary(st.session_state.document_chunks)
                st.subheader("Document Summary:")
                st.info(st.session_state.summary)
            else:
                st.error("Failed to process document. Please ensure it's a valid PDF or TXT.")

    if st.session_state.vector_store:
        st.subheader("Choose Interaction Mode:")
        mode = st.radio("Select a mode:", ("Ask Anything", "Challenge Me"))

        if mode == "Ask Anything":
            st.subheader("Ask Anything about the Document")
            question = st.text_input("Your question:")
            if st.button("Get Answer"):
                if question:
                    with st.spinner("Thinking..."):
                        answer = ask_question(question, st.session_state.vector_store)
                        st.markdown(f"**Answer:** {answer}")
                else:
                    st.warning("Please enter a question.")

        elif mode == "Challenge Me":
            st.subheader("Challenge Me: Logic-Based Questions")
           
            if st.button("Generate New Questions") or "challenge_questions" not in st.session_state:
                with st.spinner("Generating questions..."):
                    st.session_state.challenge_questions = generate_challenge_questions(st.session_state.document_chunks)
                   
                    for i, _ in enumerate(st.session_state.challenge_questions):
                         if f"user_answer_{i}" in st.session_state:
                             del st.session_state[f"user_answer_{i}"]
            
            if "challenge_questions" in st.session_state and st.session_state.challenge_questions:
                st.write("Here are three questions for you:")
                for i, q in enumerate(st.session_state.challenge_questions):
                    st.write(f"**Q{i+1}:** {q}")
                    st.text_area(f"Your answer for Q{i+1}:", key=f"user_answer_{i}")
                
            
                if st.button("Evaluate Answers", key="evaluate_answers_button_final"):
                    for i, q in enumerate(st.session_state.challenge_questions):
                        user_answer = st.session_state.get(f"user_answer_{i}", "")
                        if user_answer:
                            st.write(f"--- Evaluation for Q{i+1}:")
                            with st.spinner(f"Evaluating Q{i+1}'s answer..."):
                                feedback = evaluate_answer(q, user_answer, st.session_state.vector_store)
                                st.markdown(f"**Feedback:** {feedback}")
                        else:
                            st.warning(f"Please provide an answer for Q{i+1} to evaluate it.")
            elif uploaded_file is not None and not st.session_state.challenge_questions:
                st.info("Could not generate questions from the document. It might be too short or lack sufficient content.")