from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
import os

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """read all pdf files"""    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text(text):
    """split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

 
def get_embeddings_vector_db(chunks):
    """get embeddings for each chunk and save them into a vector database"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    vector_db = FAISS.from_texts(chunks, embedding=embeddings)
    vector_db.save_local("faiss_index")


def fetch_answer_from_llm(docs, user_question):
    """Fetches relevant answer from LLM"""

    prompt_template = """
    Answer the question as detailed as possible from the provided context, don't provide the wrong answer if not sure \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    llm_model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=llm_model, chain_type="stuff", prompt=prompt)
    return chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )


def get_similar_docs(user_question):
    """Fetches similar text from the vector db"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    return vector_db.similarity_search(user_question)


def generate_answer(user_question):
    """Gets an answer to the user query"""
    docs = get_similar_docs(user_question)
    response = fetch_answer_from_llm(docs, user_question)
    print(response)
    return response


def main():
    # Sidebar for uploading PDF files
    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit Button", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = split_text(raw_text)
                get_embeddings_vector_db(text_chunks)
                st.success("Done,  Now you can start adding your questions")

    # Main content area for displaying chat messages
    st.title("PDF ChatBot")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload your pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_answer(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
