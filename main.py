import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question  # Ensure this module is available

# Set the working directory where PDFs will be stored
working_dir = r"D:\Taha\RAG BOT GROQ\pdfs"

# Ensure directory exists
os.makedirs(working_dir, exist_ok=True)

st.title("🐋 DeepSeek-R1 - Document RAG")

# File uploader widget
uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Define save path
    save_path = os.path.join(working_dir, uploaded_file.name)
    
    # Save the file
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

        # Process the document with progress bar
        progress = st.progress(0)  # Initialize progress bar

        try:
            st.info("🔄 Processing document... Please wait.")
            progress.progress(50)  # Simulate progress halfway
            process_document_to_chroma_db(working_dir)  # Process all files in the directory
            progress.progress(100)  # Complete progress
            st.success("✅ Document processed successfully! Ready for Q&A.")
        except Exception as e:
            progress.progress(0)
            st.error(f"⚠️ Error processing document: {e}")
    
    except Exception as e:
        st.error(f"⚠️ Error saving file: {e}")

# Text input for user's question
user_question = st.text_area("💬 Ask your question about the document")

if st.button("🔍 Get Answer"):
    if user_question.strip():
        try:
            answer = answer_question(user_question)
            st.markdown("### 🤖 DeepSeek-R1 Response:")
            st.markdown(f"**{answer}**")
        except Exception as e:
            st.error(f"⚠️ Error generating answer: {e}")
    else:
        st.warning("⚠️ Please enter a question before clicking 'Get Answer'.")
