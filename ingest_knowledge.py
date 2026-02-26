from pathlib import Path
import arxiv
import yaml
from git import Repo
"""script que toma artigos (via arxiv) e repositórios para 'ingestão' de conhecimento."""

ARTICLES_DIR = Path("knowledge/articles")
REPOS_DIR = Path("knowledge/repos")

ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
REPOS_DIR.mkdir(parents=True, exist_ok=True)

def load_configs(path="source_of_all_knowledge.yml"): 
    with open(path, "r") as file: 
        return yaml.safe_load(file)


def download_articles(title: str):
    client = arxiv.Client()

    search = arxiv.Search(
        query=title,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )

    for result in client.results(search):
        paper_id = result.get_short_id()
        destiny = ARTICLES_DIR / f"{paper_id}.pdf"

        if not destiny.exists():
            result.download_pdf(dirpath=ARTICLES_DIR, filename=f"{paper_id}.pdf")
        return destiny

    raise ValueError("Artigo não encontrado")


def clone_repo(repo_url: str):
    repo_name = repo_url.rstrip("/").split("/")[-1]
    destiny = REPOS_DIR / repo_name

    if not destiny.exists():
        Repo.clone_from(repo_url, destiny)

    return destiny


def main(): 
    configs=load_configs()
    for article in configs.get('articles', []):
        download_articles(article)
    for repo in configs.get('repos', []): 
        clone_repo(repo)




if __name__ == "__main__": 

    main()
