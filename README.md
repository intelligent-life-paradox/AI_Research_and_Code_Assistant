---
title: Diff Crew Agent
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---



# Diff Crew Agent

## Assistente de Machine Learning com CrewAI

Sistema multi-agente construído com CrewAI, LlamaIndex e ChromaDB para
responder perguntas teóricas e de implementação sobre modelos de difusão
(meu tópico de pesquisa recente).

Funcionalidades novas (03/02/26): O sistema agora está encapsulado em
uma aplicação Docker com interface gráfica via Gradio.

O sistema é capaz de responder perguntas teóricas e gerar código sobre
Modelos de Difusão, utilizando uma base de conhecimento local (RAG) que
pode ser atualizada dinamicamente via interface. O sistema consulta
artigos acadêmicos e repositórios de código indexados localmente, sem
depender de fontes externas. Apesar dos agentes serem alimentados, via
RAG, por correlatos desse tópico, nada lhe impede de fazer embedding de
seus próprios arquivos ou repositórios de código.

------------------------------------------------------------------------

# Estrutura do Projeto

``` text
diff_crew/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── rag/
│   ├── pipeline_manager.py
│   └── storage/
├── crew_diffusion/
│   ├── knowledge/
│   │   ├── articles/
│   │   └── repos/
│   ├── src/
│   │   └── crew_diffusion/
│   │       ├── config/
│   │       ├── tools/
│   │       ├── crew.py
│   │       └── main.py
│   └── source_of_all_knowledge.yml
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env
```

------------------------------------------------------------------------

# Agentes do Sistema

## 1. Router Agent (Local)

Responsável por:

-   interpretar a intenção do usuário
-   delegar a tarefa ao agente correto

Características:

- Não responde perguntas. Ele só delega.

## 2. Explainer Agent

Responsável por:

-   explicações conceituais
-   fundamentos matemáticos
-   interpretação de artigos científicos

Fonte de dados:

-   coleção `articles` do ChromaDB

Restrição: nunca gera código.

## 3. Coder Agent

Responsável por:

-   geração de código Python
-   explicação de implementações
-   análise de pipelines

Fonte de dados:

-   coleção `codes` do ChromaDB

------------------------------------------------------------------------

# Tools (Isolamento Intencional)

  Tool                  Coleção ChromaDB   Agente
  --------------------- ------------------ -----------------
  articles_query_tool   articles           Explainer Agent
  codes_query_tool      codes              Coder Agent

Cada tool é isolada intencionalmente --- boa prática em sistemas
multi-agentes.

------------------------------------------------------------------------

# Configuração Inicial (Docker --- Recomendado)

## 1. Configure o arquivo .env

``` bash
GROQ_API_KEY=gsk_sua_chave_aqui
```

## 2. Suba o container

``` bash
docker-compose up --build
```

Acesse:

    http://localhost:7860

------------------------------------------------------------------------

# RAG Dinâmico

A interface permite:

-   upload de PDFs
-   ingestão de repositórios GitHub
-   reindexação em tempo real

------------------------------------------------------------------------

# Pipeline Manual (Legado)

## Passo I --- Editar fontes

Arquivo `source_of_all_knowledge.yml`:

``` yaml
articles:
  - https://arxiv.org/pdf/2006.11239

repos:
  - https://github.com/huggingface/diffusers
```

## Passo II --- Indexação

``` bash
python rag/pipeline_manager.py --mode all
```

## Passo III --- Executar Crew

``` bash
cd crew_diffusion/src
python -m crew_diffusion.main
```

------------------------------------------------------------------------

# Funcionalidades Atualizadas em 03/02/26

## Full Docker Support

Aplicação completamente containerizada.

## Interface Gradio

Interação e gerenciamento via navegador.



## Pipeline Unificado

Centralização da ingestão em:

``` bash
rag/pipeline_manager.py
```

## Pipeline Automatizado (GitHub Actions)

Build Docker e testes automáticos a cada push.

------------------------------------------------------------------------

# Observações Importantes

## 1. Gargalo de API

Uso apenas de APIs gratuitas. Possível ocorrência de RateLimit.

Recomendações:

-   reduzir temperatura dos modelos
-   alterar modelos em crew.py

## 2. Estado do Projeto

Projeto em desenvolvimento contínuo, funcionando como complemento de um
sistema maior de pesquisa em Machine Learning.

A atualização de 03/02/26 focou em:

-   portabilidade
-   usabilidade
-   reprodutibilidade científica

------------------------------------------------------------------------

# Resumo

O Diff Crew Agent transforma artigos e repositórios em um assistente de
pesquisa interativo, combinando:

-   RAG local
-   agentes especializados
-   geração de código
-   explicações teóricas profundas
-   execução containerizada

Um passo em direção a assistentes científicos especializados e
reproduzíveis.
