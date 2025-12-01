

import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import initialize_agent, AgentType


# ----------------------------------------------------
# UI - Title
# ----------------------------------------------------
st.title("🔎 AI Assistant with Groq + Tavily + Arxiv + Wikipedia")


# ----------------------------------------------------
# Sidebar API Keys
# ----------------------------------------------------
st.sidebar.header("🔐 API Keys")
groq_key = st.sidebar.text_input("Groq API Key:", type="password")
tavily_key = st.sidebar.text_input("Tavily API Key:", type="password")

# Set environment var for Tavily
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key


# ----------------------------------------------------
# Tools: Tavily, ArXiv, Wikipedia
# ----------------------------------------------------
arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=2000
    )
)

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=2000
    )
)

tavily_tool = TavilySearchResults(k=5)

tools = [tavily_tool, arxiv_tool, wiki_tool]


# ----------------------------------------------------
# Chat History
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I can search the web, Arxiv, Wikipedia and answer anything. How can I help you?"}
    ]


# Display messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ----------------------------------------------------
# Chat Input
# ----------------------------------------------------
prompt = st.chat_input("Ask me anything...")

if prompt and groq_key and tavily_key:

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Setup LLM
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.1-8b-instant",
        streaming=True
    )

    # Create agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=5,
        verbose=True,
        handle_parsing_errors=True
    )

    # Assistant response with streaming
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        output = agent.run(prompt, callbacks=[st_cb])

        # Save and display result
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.write(output)


elif prompt and not groq_key:
    st.warning("Please enter your Groq API key.")

elif prompt and not tavily_key:
    st.warning("Please enter your Tavily API key.")
