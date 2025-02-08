import os
import json
from PyPDF2 import PdfReader  # Replacing UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Setup working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load configuration safely
config_path = os.path.join(working_dir, "config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError("⚠️ config.json not found in the working directory.")

with open(config_path, "r") as config_file:
    config_data = json.load(config_file)

# Check for GROQ_API_KEY
GROQ_API_KEY = config_data.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY is missing in config.json.")

# Set environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load embedding model
embedding = HuggingFaceEmbeddings()

# Initialize LLM from Groq
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0
)


def extract_text_from_pdf(file_path):
    """
    Extract text content from a PDF file using PyPDF2.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"⚠️ Error extracting text from PDF: {e}")


def process_document_to_chroma_db(directory_path):
    """
    Process all PDF documents in the given directory, split their text, 
    and store embeddings in a persistent ChromaDB.
    """
    try:
        # Iterate through all PDF files in the directory
        for file_name in os.listdir(directory_path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(directory_path, file_name)
                print(f"📂 Processing document: {file_name}")

                # Extract text from the PDF
                text = extract_text_from_pdf(file_path)
                if not text.strip():
                    raise ValueError(f"⚠️ No text extracted from '{file_name}'. The file might be empty.")

                # Split text into chunks
                print("🔄 Splitting document into smaller chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_text(text)

                # Create a persistent ChromaDB instance
                print("💾 Storing embeddings in ChromaDB...")
                vectordb = Chroma.from_texts(
                    texts=texts,
                    embedding=embedding,
                    persist_directory=os.path.join(working_dir, "doc_vectorstore")
                )

        print("✅ All documents successfully processed and stored in ChromaDB.")
        return "✅ Documents successfully processed and stored in ChromaDB."

    except Exception as e:
        raise RuntimeError(f"⚠️ Error processing documents: {e}")


def answer_question(user_question):
    """
    Retrieve and generate an answer for the given user question 
    based on the stored document embeddings.
    """
    try:
        # Load the persistent vector database
        vectordb_path = os.path.join(working_dir, "doc_vectorstore")
        if not os.path.exists(vectordb_path):
            raise FileNotFoundError("⚠️ ChromaDB vector store not found. Please process a document first.")

        print("📂 Loading vector database...")
        vectordb = Chroma(
            persist_directory=vectordb_path,
            embedding_function=embedding
        )

        # Create a retriever from the vector database
        retriever = vectordb.as_retriever()

        # Create a QA chain with DeepSeek-R1
        print("🤖 Initializing Retrieval QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )

        # Invoke the QA chain with the user question
        print("💬 Generating answer...")
        response = qa_chain.invoke({"query": user_question})
        answer = response.get("result", "⚠️ No response generated.")

        return answer

    except Exception as e:
        raise RuntimeError(f"⚠️ Error generating response: {e}")
