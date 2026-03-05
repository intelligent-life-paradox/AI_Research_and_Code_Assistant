import gradio as gr
import os
import sys
from litellm import completion
import time
import random

# Adiciona o src ao path para importar o crew corretamente
sys.path.append(os.path.join(os.getcwd(), 'crew_diffusion', 'src'))

from crew_diffusion.main import run as run_crew_logic
from rag.pipeline_manager import download_article_by_title, clone_repo_from_url, save_uploaded_file, run_indexing_process

def router_agent(query):
    """
    Agente Router simples usando Groq/Llama3.
    """
    prompt = f"""
    Classifique a seguinte pergunta sobre Modelos de Difusão em uma das categorias:
    1. 'Explainer Agent' (Perguntas teóricas, matemáticas, conceitos, artigos).
    2. 'Coder Agent' (Implementação, código Python, erros, bibliotecas).
    
    Pergunta: "{query}"
    
    Responda APENAS com o nome do agente.
    """
    model = os.getenv("MODEL_ROUTER", "openrouter/qwen/qwen2.5-7b-instruct")
    api_key = os.getenv("OPENROUTER_API_KEY")
    api_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    max_tokens = int(os.getenv("MAX_TOKENS_ROUTER", "16"))
    retry_attempts = int(os.getenv("RETRY_ATTEMPTS", "3"))
    backoff_base = float(os.getenv("RETRY_BACKOFF_BASE_SECONDS", "0.8"))

    for attempt in range(retry_attempts):
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                api_base=api_base,
                max_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == retry_attempts - 1:
                return "Indefinido (Erro API)"
            sleep_s = (backoff_base * (2 ** attempt)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)

def handle_chat(query, mode):
    if not query:
        return "Por favor, digite uma pergunta."
    
    classification = router_agent(query)
    
    if mode == "Apenas Classificar":
        return f"🤖 **Router Decision:** Essa tarefa deve ser enviada para o **{classification}**."
    
    
     return f"⏳ **Processando...**\n\n" + \
           run_crew_logic(user_query=query)

def handle_training(files, arxiv_title, repo_url):
    logs = []
    
    
    if files:
        for f in files:
            logs.append(save_uploaded_file(f))
            
    
    if arxiv_title and arxiv_title.strip():
        logs.append(download_article_by_title(arxiv_title))
        
    
    if repo_url and repo_url.strip():
        urls = [u.strip() for u in repo_url.split(',')]
        for url in urls:
            logs.append(clone_repo_from_url(url))
            
    
    logs.append("\n Iniciando Indexação no ChromaDB...")
    logs.append(run_indexing_process())
    
    return "\n".join(logs)


with gr.Blocks(title="Diff Crew Lab", theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# Diff Crew — Local AI Lab")
    
    with gr.Tabs():
        
        with gr.Tab("💬  Chat & Agentes "):
            with gr.Row():
                inp = gr.Textbox(label="Pergunta", placeholder="Ex: Explique a matemática do DDPM ou Crie um script de UNet")
            with gr.Row():
                btn_class = gr.Button("🔍 Classificar Agente", variant="secondary")
                btn_run = gr.Button("🚀 Executar Crew", variant="primary")
            out = gr.Markdown(label="Resposta")
            
            btn_class.click(fn=handle_chat, inputs=[inp, gr.State("Apenas Classificar")], outputs=out)
            btn_run.click(fn=handle_chat, inputs=[inp, gr.State("Executar")], outputs=out)

        #
        with gr.Tab("📚 Base de Conhecimento"):
            gr.Markdown("Adicione conhecimento. Os arquivos serão salvos em `knowledge/` e indexados no ChromaDB.")
            
            with gr.Row():
                file_up = gr.File(label="Upload PDF Local", file_count="multiple", file_types=[".pdf"])
                arxiv_in = gr.Textbox(label="Título no Arxiv (Baixar via API)", placeholder="Ex: Attention is all you need")
                repo_in = gr.Textbox(label="URL Git (Clonar)", placeholder="https://github.com/user/repo")
            
            btn_train = gr.Button("💾 Ingerir e Treinar", variant="stop")
            log_out = gr.Textbox(label="Logs do Sistema", lines=10)
            
            btn_train.click(fn=handle_training, inputs=[file_up, arxiv_in, repo_in], outputs=log_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)