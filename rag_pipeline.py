from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class RAGPipeline:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

        # Embedding model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.chunks = []
        self.index = None

    # -----------------------------------
    # Extract text from PDF
    # -----------------------------------

    def extract_text(self):

        reader = PdfReader(self.pdf_path)

        text = ""

        for page in reader.pages:
            text += page.extract_text()

        return text

    # -----------------------------------
    # Split text into chunks
    # -----------------------------------

    def chunk_text(
        self,
        text,
        chunk_size=500,
        overlap=50
    ):

        chunks = []

        for i in range(
            0,
            len(text),
            chunk_size - overlap
        ):

            chunk = text[i:i + chunk_size]

            chunks.append(chunk)

        self.chunks = chunks

        return chunks

    # -----------------------------------
    # Create embeddings
    # -----------------------------------

    def create_embeddings(self):

        embeddings = self.model.encode(
            self.chunks
        )

        return np.array(
            embeddings
        ).astype("float32")

    # -----------------------------------
    # Build FAISS Index
    # -----------------------------------

    def build_faiss_index(
        self,
        embeddings
    ):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(embeddings)

    # -----------------------------------
    # Retrieve relevant chunks
    # -----------------------------------

    def retrieve(
        self,
        query,
        top_k=3
    ):

        query_embedding = self.model.encode(
            [query]
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        retrieved_chunks = []

        for idx in indices[0]:

            retrieved_chunks.append(
                self.chunks[idx]
            )

        return retrieved_chunks

    # -----------------------------------
    # Setup complete pipeline
    # -----------------------------------

    def setup(self):

        # Extract text
        text = self.extract_text()

        # Chunking
        self.chunk_text(text)

        # Embeddings
        embeddings = self.create_embeddings()

        # FAISS
        self.build_faiss_index(
            embeddings
        )