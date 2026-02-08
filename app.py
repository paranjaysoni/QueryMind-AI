import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="QueryMind AI", page_icon="🤖")

st.title("QueryMind AI: GenAI Chatbot")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

# LangSmith Tracking
# LangSmith Tracking
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Project1-GenAIChatbot"
    os.environ["LANGSMITH_PROJECT"] = "Project1-GenAIChatbot"

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

model_name = st.sidebar.selectbox(
    "Select Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
)

# LLM
llm = ChatGroq(
    model=model_name,
    temperature=temperature,
    groq_api_key=GROQ_API_KEY
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Give clear and detailed answers."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | llm

# Session State Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)

# User Input
user_input = st.chat_input("Ask anything...")

if user_input:

    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream({"messages": st.session_state.chat_history}):
            full_response += chunk.content
            response_placeholder.write(full_response)

    st.session_state.chat_history.append(AIMessage(content=full_response))

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Footer Credit
st.markdown("---")
st.markdown(
    "<h5 style='text-align: center; color: grey;'>Built with  ❤️  by <a href='https://github.com/paranjaysoni' target='_blank' style='color: grey; text-decoration: none;'>Paranjay Soni</a></h5>",
    unsafe_allow_html=True
)