import os
import sys
import shutil
import argparse
import arxiv
import yaml
from pathlib import Path
from git import Repo
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large",
    device="cpu",
    embed_batch_size=2
)
Settings.llm = None

# Caminhos
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_DIR = BASE_DIR / "crew_diffusion" / "knowledge"
ARTICLES_DIR = KNOWLEDGE_DIR / "articles"
REPOS_DIR = KNOWLEDGE_DIR / "repos"
CHROMA_PATH = BASE_DIR / "rag" / "storage" / "chroma_db"
CONFIG_PATH = BASE_DIR / "source_of_all_knowledge.yml"


def load_configs():
    if not CONFIG_PATH.exists():
        print(f"[WARNING] Config file not found at {CONFIG_PATH}")
        return {}
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)

def download_article_by_title(title: str):
    print(f"[INFO] Searching for article: {title}")
    client = arxiv.Client()
    search = arxiv.Search(query=title, max_results=1, sort_by=arxiv.SortCriterion.Relevance)
    
    results = list(client.results(search))
    if not results:
        print(f"error: Article '{title}' not found.")
        return

    result = results[0]
    filename = f"{result.get_short_id()}.pdf"
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    
    result.download_pdf(dirpath=str(ARTICLES_DIR), filename=filename)
    print(f" Downloaded: {result.title}")

def clone_repo_from_url(repo_url: str):
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    destiny = REPOS_DIR / repo_name
    REPOS_DIR.mkdir(parents=True, exist_ok=True)

    if destiny.exists():
        print(f"[WARNING] Repo {repo_name} already exists.")
        return
    
    try:
        print(f"[INFO] Cloning {repo_name}...")
        Repo.clone_from(repo_url, destiny)
        print(f"[SUCCESS] Cloned {repo_name}.")
    except Exception as e:
        print(f"[ERROR] Error cloning {repo_name}: {e}")

def save_uploaded_file(file_obj):
    """Salva arquivo PDF enviado via Upload direto (Gradio)."""
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = ARTICLES_DIR / os.path.basename(file_obj.name)
    shutil.copy(file_obj.name, dest_path)
    return f"[SUCCESS] File saved: {os.path.basename(file_obj.name)}"

def run_indexing_process():
    print("\n[INFO] Starting Indexing Process...")
    try:
        CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        db = chromadb.PersistentClient(path=str(CHROMA_PATH))
        
        collection_articles = db.get_or_create_collection("articles")
        vector_store_articles = ChromaVectorStore(chroma_collection=collection_articles)
        storage_context_articles = StorageContext.from_defaults(vector_store=vector_store_articles)
        
        if ARTICLES_DIR.exists() and any(ARTICLES_DIR.iterdir()):
            docs_pdf = SimpleDirectoryReader(str(ARTICLES_DIR)).load_data()
            VectorStoreIndex.from_documents(docs_pdf, storage_context=storage_context_articles)
            print(f"[INFO] Indexed {len(docs_pdf)} article pages.")
        
        collection_codes = db.get_or_create_collection("codes")
        vector_store_codes = ChromaVectorStore(chroma_collection=collection_codes)
        storage_context_codes = StorageContext.from_defaults(vector_store=vector_store_codes)

        if REPOS_DIR.exists() and any(REPOS_DIR.iterdir()):
            exclude = [".venv", "__pycache__", ".git", "build", "dist", "site-packages"]
            
            py_docs = SimpleDirectoryReader(
                str(REPOS_DIR), recursive=True, required_exts=[".py"], exclude=exclude
            ).load_data()
            
            text_docs = SimpleDirectoryReader(
                str(REPOS_DIR), recursive=True, required_exts=[".md", ".yml"], exclude=exclude
            ).load_data()

            py_splitter = CodeSplitter(language="python", chunk_lines=120, chunk_lines_overlap=20)
            text_splitter = SentenceSplitter()

            nodes = py_splitter.get_nodes_from_documents(py_docs) + \
                    text_splitter.get_nodes_from_documents(text_docs)
            
            VectorStoreIndex(nodes, storage_context=storage_context_codes)
            print(f"[INFO] Indexed {len(nodes)} code/text nodes.")
            
        return "Indexing Complete"

    except Exception as e:
        print(f" Fatal error during indexing: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ingest", "index", "all"], default="all", help="Operation mode")
    args = parser.parse_args()

    if args.mode in ["ingest", "all"]:
        print("STEP 1:  IT'S INGESTION TIME! ")
        configs = load_configs()
        
        for article in configs.get('articles', []):
            download_article_by_title(article)
            
        for repo in configs.get('repos', []):
            clone_repo_from_url(repo)

    if args.mode in ["index", "all"]:
        print("\n STEP 2: INDEXING TIME! ")
        run_indexing_process()