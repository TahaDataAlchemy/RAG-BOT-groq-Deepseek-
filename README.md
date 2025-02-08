![image](https://github.com/user-attachments/assets/bff04903-ae75-46cb-8e95-7bd2bbbe7a7f)

DeepSeek-R1 - Document RAG Bot

Overview
DeepSeek-R1 is a powerful Retrieval-Augmented Generation (RAG) bot that allows users to upload PDF documents, ask questions about their content, and receive accurate answers powered by DeepSeek AI.

Features

PDF Upload: Users can upload PDF files for processing.
Question Answering: Ask questions about the uploaded document and receive precise responses.
Efficient Document Processing: Splits and stores document content for quick retrieval.
Powered by AI: Utilizes advanced language models like DeepSeek-R1 for response generation.

Requirements
Python 3.8 or higher
Libraries:
streamlit
PyPDF2
langchain
torch
chromadb
langchain-text-splitters


How to Use

Open the app in your browser (Streamlit will provide the local URL).
Upload a PDF file.
Wait for the document to be processed.
Enter your question in the text box and click "Get Answer."
View the AI-generated response.

Project Structure

main.py: Streamlit interface for user interactions.
rag_utility.py: Handles PDF processing, vector storage, and question answering.
config.json: Contains the API key for DeepSeek.
Example Workflow
Upload a file: human_cells.pdf.
Ask a question: "What are the main functions of human cells?"
Receive an answer: "Human cells perform functions like energy production, protein synthesis, and waste elimination."




