import streamlit as st

from rag_pipeline import RAGPipeline
from llm import LLM

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="AI Academic Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------
# Session State
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag" not in st.session_state:
    st.session_state.rag = None

if "llm" not in st.session_state:
    st.session_state.llm = LLM()

# -----------------------------------
# Header
# -----------------------------------

st.markdown("""
# 🤖 AI Academic Assistant

Ask questions from your uploaded PDF documents using
**Retrieval-Augmented Generation (RAG)**,
**Semantic Search**, and **Large Language Models**.
""")

st.markdown("---")

# -----------------------------------
# Sidebar
# -----------------------------------

with st.sidebar:

    st.header("📚 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Processing document..."):

            rag = RAGPipeline("temp.pdf")
            rag.setup()

            st.session_state.rag = rag

        st.success("Document processed successfully ✅")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
        This chatbot uses:

        - FAISS Vector Database
        - Sentence Transformers
        - Retrieval-Augmented Generation
        - Large Language Models
        """
    )

# -----------------------------------
# Chat History
# -----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# -----------------------------------
# User Query
# -----------------------------------

query = st.chat_input(
    "Ask a question about the uploaded document..."
)

# -----------------------------------
# Generate Response
# -----------------------------------

if query:

    if st.session_state.rag is None:

        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Searching document..."):

            retrieved = (
                st.session_state.rag.retrieve(query)
            )

            context = " ".join(retrieved)

            answer = (
                st.session_state.llm.generate_answer(
                    query,
                    context
                )
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.write(answer)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Retrieved Chunks",
                len(retrieved)
            )

        with col2:
            st.metric(
                "Document Status",
                "Ready"
            )

        with st.expander(
            "📄 View Retrieved Context"
        ):

            for i, chunk in enumerate(retrieved):

                st.markdown(
                    f"### Source {i+1}"
                )

                st.info(chunk)

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.caption(
    "Built with Streamlit • FAISS • SentenceTransformers • LLMs"
)