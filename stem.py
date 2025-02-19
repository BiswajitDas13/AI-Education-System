import streamlit as st
import openai
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import re

# OpenAI API Key (Replace with your own key)
openai.api_key = "Your openai APi key"

# Bot options
bots = {
    "Class 11 Math": "Math_11_CBSC.pdf",
    "Class 12 Math": "Math_12_CBSC.pdf",
    "Class 11 Physics": "Physics_11_CBSC.pdf",
    "Class 12 Physics": "Physics_12_CBSC.pdf"
}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf = PdfReader(pdf_path)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

# Convert text to LangChain Document format
def text_to_docs(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    chunks = text_splitter.split_text(" ".join(text))
    return [Document(page_content=chunk, metadata={"source": f"page-{i+1}"}) for i, chunk in enumerate(chunks)]

# Create FAISS vector database
def create_vectordb(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    docs = text_to_docs(text)
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    return FAISS.from_documents(docs, embeddings)

# Streamlit UI
st.set_page_config(page_title="Subject Bot", layout="wide")
st.title("📚 AI Education System")

# Footer dropdown to select bot
selected_bot = st.selectbox("Select a Bot", list(bots.keys()), key="bot_selector")

# Load vector database based on selected bot
vectordb = create_vectordb(bots[selected_bot])

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.chat_message(chat["role"]).write(chat["content"])

# User input
user_input = st.text_input("Ask question", key="user_query")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    # Retrieve relevant content
    search_results = vectordb.similarity_search(user_input, k=3)
    pdf_extract = "\n".join([res.page_content for res in search_results])
    
    system_prompt = f"""
    You are an AI tutor for {selected_bot}. Answer user questions based on the given context. 
    Reply 'Not applicable' if it's outside of the subject.
    you are expert on the math and physics based {selected_bot} you will replying all the questions of that subject.
    if anyone asked question in bengali you will reply in bengali
    Do not hallucinate provind answer based on pdf context and outside of choice subjects.
    \n\nPDF Content:\n{pdf_extract}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    )
    
    bot_reply = response["choices"][0]["message"]["content"]
    
    with st.chat_message("assistant"):
        st.write(bot_reply)
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
