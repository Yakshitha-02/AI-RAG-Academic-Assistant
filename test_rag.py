from rag_pipeline import RAGPipeline
from llm import LLM


# -----------------------------------
# Initialize RAG Pipeline
# -----------------------------------

rag = RAGPipeline("sample.pdf")


# -----------------------------------
# Setup Pipeline
# -----------------------------------

rag.setup()


# -----------------------------------
# Initialize LLM
# -----------------------------------

llm = LLM()


# -----------------------------------
# Chat Loop
# -----------------------------------

while True:

    query = input(
        "\nAsk a question (type 'exit' to quit): "
    )

    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    retrieved = rag.retrieve(query)

    print("\nRelevant Sections:\n")

    for i, chunk in enumerate(retrieved):

        print(f"Chunk {i+1}:\n")

        print(chunk)

        print("\n" + "-" * 50 + "\n")

    # Combine chunks into context
    context = " ".join(retrieved)

    # Generate final answer
    answer = llm.generate_answer(
        query,
        context
    )

    print("\nFinal Answer:\n")

    print(answer)