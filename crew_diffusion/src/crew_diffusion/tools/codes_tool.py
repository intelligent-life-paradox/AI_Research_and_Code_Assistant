import chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
    embed_batch_size=2
)
Settings.llm = None

CHROMA_PATH = Path(__file__).parent.parent.parent / "rag" / "storage" / "chroma_db"

class CodesQueryInput(BaseModel):
    query: str = Field(..., description="Code-related question about diffusion model implementations")

class CodesQueryTool(BaseTool):
    name: str = "codes_query_tool"
    description: str = ("""
        Queries the 'codes' ChromaDB collection containing source code repositories 
        related to diffusion models. Use this to retrieve implementation examples, 
        function signatures, and code patterns. Returns relevant code snippets."""
    )
    args_schema: type[BaseModel] = CodesQueryInput

    def _run(self, query: str) -> str:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        chroma_collection = chroma_client.get_collection("codes")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        query_engine = index.as_query_engine(similarity_top_k=5)
        response = query_engine.query(query)
        return str(response)