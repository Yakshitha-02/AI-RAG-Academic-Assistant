import streamlit as st

from rag_pipeline import RAGPipeline
from llm import LLM


# -----------------------------------
# Title
# -----------------------------------

st.title("RAG Chatbot")


# -----------------------------------
# Upload PDF
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)


# -----------------------------------
# Process PDF
# -----------------------------------

if uploaded_file is not None:

    # Save uploaded PDF
    with open("temp.pdf", "wb") as f:

        f.write(uploaded_file.read())

    st.success(
        "PDF uploaded successfully!"
    )

    # Initialize RAG Pipeline
    rag = RAGPipeline("temp.pdf")

    # Setup Pipeline
    rag.setup()

    # Initialize LLM
    llm = LLM()

    # User Query
    query = st.text_input(
        "Ask a question"
    )

    # Generate Response
    if query:

        # Retrieve chunks
        retrieved = rag.retrieve(query)

        # Combine context
        context = " ".join(retrieved)

        # Generate answer
        answer = llm.generate_answer(
            query,
            context
        )

        # Display answer
        st.subheader("Answer")

        st.write(answer)

        # Display retrieved chunks
    with st.expander("📄 View Retrieved Context"):

     for i, chunk in enumerate(retrieved):

        st.markdown(f"### Chunk {i+1}")

        st.write(chunk)

        st.write("---")