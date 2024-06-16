import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI

def get_pdf_text(pdf_docs):
    text = ""
    for p in pdf_docs:
        pdf_reader = PdfReader(p)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_pdf_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator = '\n', chunk_size = 1000, chunk_overlap = 100, length_function= len)
    chunks = text_splitter.split_text(raw_text)
    return chunks



def get_vector_storage(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    # embeddings = OpenAIEmbeddings()
    
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore


def get_converse_chain(vectorstore):
    llm = ChatOpenAI()
    
    llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs = {'temparature':0.5, 'max_length':512})
    
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vectorstore.as_retriever(),memory=memory)
    return conversation_chain


def handle_user_input(user_q):
    response = st.session_state.conversation({"question":user_q})
    # st.write(response)
    st.session_state.chat_history = response["chat_history"]
    
    for i, message in enumerate[st.session_state.chat_history]:
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)
            
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiple PDF's :books:")    
    user_question = st.text_input("Ask Question about your documents:")
    if user_question:
        handle_user_input(user_question)
    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs = st.file_uploader("Upload your PDFs here and press process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_t = get_pdf_text(pdf_docs)
                
                text_chunks = get_pdf_chunks(raw_t)

                vectorstore = get_vector_storage(text_chunks)
                
                st.session_state.conversation = get_converse_chain(vectorstore)

if __name__ == "__main__":
    main()