import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Use the Hugging Face token directly
hf_token = "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"

# Check if HF_TOKEN is not None
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
else:
    st.error("HF_TOKEN is not set. Please provide a valid HuggingFace token.")

# Use the default Groq API Key
default_groq_api_key = "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"

# Initialize Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit interface
st.title("Conversational PDF Assistant with Chat History")
st.write("Upload PDFs and chat with their content")

# Initialize the Groq model using the default API key
llm = ChatGroq(groq_api_key=default_groq_api_key, model_name="Gemma2-9b-It")

# Chat interface
session_id = st.text_input("Session ID", value="default_session")

# Statefully manage chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

# Process uploaded PDFs
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Initialize FAISS with the documents
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)

        # Show loading indicator
        with st.spinner("Thinking..."):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },
            )

        # Display the assistant's response with increased font size for "Assistant:"
        st.markdown("<h3 style='font-size: 20px;'>Assistant:</h3>", unsafe_allow_html=True)
        st.write(response['answer'])
        st.write("")
        st.write("")

        # Display the chat history with increased font size
        st.markdown("<h3 style='font-size: 20px;'>Chat History:</h3>", unsafe_allow_html=True)

        # Reverse the chat history for display
        for message in reversed(session_history.messages):
            st.markdown(f"<h4 style='font-size: 18px;'>{message.__class__.__name__}:</h4>", unsafe_allow_html=True)
            st.write(message.content)
