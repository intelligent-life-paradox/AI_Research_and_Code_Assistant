# 🤖 Diff Crew — Assistente de Machine Learning com CrewAI

Sistema multi-agente construído com **CrewAI**, **LlamaIndex** e **ChromaDB** para responder perguntas teóricas e de implementação sobre modelos de difusão. O sistema consulta artigos acadêmicos e repositórios de código indexados localmente, sem depender de fontes externas.

---

##  Estrutura do Projeto

```
diff_crew/
├── .github/
│   └── workflows/
│       └── pipeline.yml          # Pipeline automatizado (GitHub Actions)
├── crew_diffusion/
│                # Repositórios de código clonados
│
│   ├── src/
│   │   └── crew_diffusion/
│   │       ├── config/
│   │       │   ├── agents.yaml   # Definição de role, goal e backstory dos agentes
│   │       │   └── tasks.yaml    # Definição das tarefas de cada agente
│   │       ├── tools/
│   │       │   ├── __init__.py
│   │       │   ├── articles_tool.py  # Tool que consulta a coleção "articles" no ChromaDB
│   │       │   └── codes_tool.py     # Tool que consulta a coleção "codes" no ChromaDB
│   │       ├── crew.py           # Montagem do Crew, agentes, LLMs e injeção de tools
│   │       └── main.py           # Ponto de entrada da aplicação
├── source_of_all_knowledge.yml  # Arquivo de configuração das fontes de conhecimento
├── ingest_knowledge.py          # Script de ingestão das fontes para a pasta knowledge/
│                     
├── .env                             # Variáveis de ambiente (GROQ_API_KEY)
├── knowledge/
│      ├── articles/             # PDFs, livros e artigos sobre modelos de difusão
│      └── repos/
│requirements.txt
├── rag/
│   │   ├── index_pdfs.ipynb      # Notebook: indexa artigos no ChromaDB
│   │   └── index_codes.ipynb
        └── storage/chromadb # Banco vetorial ChromaDB (gerado automaticamente após rodar os notebooks na sequência que aparecem)
          
```

---

##  Agentes (os agentes a seguir foram feitos usando-se do template disponibilizado pelo crewai)

### 1. Manager Agent
Responsável exclusivamente por entender a intenção do usuário e delegar a tarefa para o agente correto. **Não responde perguntas técnicas diretamente.** Usa um modelo menor e mais rápido.

### 2. Explainer Agent
Responde perguntas teóricas, conceituais e matemáticas sobre modelos de difusão. Consulta **apenas** a coleção `articles` do ChromaDB (artigos e livros). Nunca gera código. 

### 3. Coder Agent
Responde perguntas sobre implementações e gera código Python. Consulta **apenas** a coleção `codes` do ChromaDB (repositórios de código). Pode salvar arquivos `.py` ou `.ipynb` na pasta `output/`.

---

##  Tools

| Tool | Coleção ChromaDB | Agente que usa |
|---|---|---|
| `articles_query_tool` | `articles` | Explainer Agent |
| `codes_query_tool` | `codes` | Coder Agent |

Cada tool é isolada intencionalmente — isso é basicamente uma boa prática quando se lida com agentes.

---

##  Configuração Inicial

### 1. Clone o repositório e ative o ambiente virtual

```bash
git clone <url-do-repositorio>
cd diff_crew/crew_diffusion
python -m venv venv
venv\Scripts\activate        # Windows
# ou
source venv/bin/activate     # Linux/Mac
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure o arquivo `.env`

```bash
GROQ_API_KEY=gsk_sua_chave_aqui *
```

---

##  Como Rodar o Pipeline Manualmente

### Passo I — Edite as fontes de conhecimento

Abra o arquivo `source_of_all_knowledge.yml` e adicione os caminhos dos seus artigos/PDFs e repositórios de código que deseja indexar:

```yaml
articles:
  - https://arxiv.org/pdf/2006.11239   # DDPM
  - https://arxiv.org/pdf/2010.02502   # Score Matching

repos:
  - https://github.com/huggingface/diffusers
  - https://github.com/openai/consistency_models
``` ACIMA SÃO SÓ EXEMPLOS(!!!)

#

Execute o script de ingestão. Ele vai baixar e salvar os arquivos na pasta `knowledge/`:

```bash
python ingest_knowledge.py
```

### Passo II — Indexe os PDFs no ChromaDB

Abra e execute o notebook **na ordem correta**:

```bash
# Primeiro: indexa artigos e livros
jupyter nbconvert --to notebook --execute rag/index_pdfs.ipynb

# Segundo: indexa repositórios de código
jupyter nbconvert --to notebook --execute rag/index_codes.ipynb
```

>  **Importante:** o modelo de embeddings usado aqui deve ser o mesmo configurado nas tools (`articles_tool.py` e `codes_tool.py`). Se mudar o modelo de embeddings, precisará re-indexar tudo.

### Passo III — Execute o Crew

```bash
cd src
python -m crew_diffusion.main
```

O sistema vai pedir uma pergunta:

```
Ask anything about diff. models or codes: Explique o processo de difusão direta no DDPM (Exemplo)
```

O Manager Agent vai classificar a pergunta e delegar ao agente correto, que consultará o ChromaDB e retornará a resposta... ou pelo menos é isso que esperamos...

---

##  Pipeline Automatizado (GitHub Actions)

O arquivo `.github/workflows/pipeline.yml` automatiza todos os passos acima a cada `push` na branch `main`, ou pode ser disparado manualmente pela aba **Actions** do GitHub.

Para usar, adicione sua chave no GitHub:
> Repositório → Settings → Secrets and variables → Actions → `GROQ_API_KEY`

---

##  Dependências Principais

| Pacote | Função |
|---|---|
| `crewai` | Orquestração dos agentes |
| `litellm` | Interface com a API do Groq |
| `llama-index` | Query engine sobre o ChromaDB |
| `chromadb` | Banco vetorial local |
| `fastembed` | Geração de embeddings (local, gratuito) |
| `python-dotenv` | Leitura do `.env` |

## Observações importantes!!!

### 1.
Um possível gargalo desse projeto é a API. Esse futuro cientista de dados que vos fala não usou nenhuma API paga, portanto, é provável que se gere o erro de RateLimit — ou algo similar. 
Você também pode — quem sabe até deve — mudar os modelos para cada agente indo em crew_diffusion\src\crew_diffusion\crew.py. Selecione os melhores modelos, deixe as temperaturas baixas para eles não delirarem nessas tarefas, salve o arquivo novamente e rode a pipeline como descrito acima.
De preferência, selecione algum modelo disponível pela API do GROQ para você não ter que mudar as chamadas de enviroment em alguns files.

### 2.
Esse projeto é um complemento de outro projeto, contudo, ainda é um projeto em andamento. Se você está lendo isso nesse momento, ainda implementarei um aplicativo para ficar algo ainda mais user-friendly, conectarei com algum framework de api e usarei docker também. 
