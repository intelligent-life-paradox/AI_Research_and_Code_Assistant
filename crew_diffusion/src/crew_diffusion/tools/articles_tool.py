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

CHROMA_PATH = Path("/app/rag/storage/chroma_db")

class ArticlesQueryInput(BaseModel):
    query: str = Field(..., description="Theoretical question about diffusion models")

class ArticlesQueryTool(BaseTool):
    name: str = "articles_query_tool"
    description: str = (
        """you will query the 'articles' ChromaDB collection containing books and academic 
        papers about diffusion models. Use this to answer theoretical, conceptual, 
        or mathematical questions. Returns relevant text passages."""
    )
    args_schema: type[BaseModel] = ArticlesQueryInput

    def _run(self, query: str) -> str:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        chroma_collection = chroma_client.get_collection("articles")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)
        return str(response)