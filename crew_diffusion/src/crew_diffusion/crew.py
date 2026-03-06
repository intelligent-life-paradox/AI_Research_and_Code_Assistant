import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileWriterTool
from crewai import LLM

from crew_diffusion.tools.articles_tool import ArticlesQueryTool
from crew_diffusion.tools.codes_tool import CodesQueryTool


@CrewBase
class CreateCrew():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def _build_llm(self, role: str, default_model: str, default_temp: float, default_max_tokens: int):
        model = os.getenv(f'MODEL_{role}', default_model)
        
        if model.startswith('groq/'):
            api_key = os.getenv('GROQ_API_KEY')
            api_base = 'https://api.groq.com/openai/v1'
        else:
            api_key = os.getenv('OPENROUTER_API_KEY')
            api_base = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')

        retries = int(os.getenv('RETRY_ATTEMPTS', '3'))
        timeout = int(os.getenv('LLM_TIMEOUT_SECONDS', '60'))

        return LLM(
            model=model,
            temperature=float(os.getenv(f'TEMP_{role}', str(default_temp))),
            max_tokens=int(os.getenv(f'MAX_TOKENS_{role}', str(default_max_tokens))),
            api_key=api_key,
            api_base=api_base,
            max_retries=retries,
            timeout=timeout,
        )

    
    def _manager_llm(self):
        return self._build_llm(
            role='MANAGER',
            default_model='openrouter/meta-llama/llama-3.1-8b-instruct',
            default_temp=0.0,
            default_max_tokens=400,
        )


    def _explainer_llm(self):
        return self._build_llm(
            role='EXPLAINER',
            default_model='openrouter/qwen/qwen2.5-14b-instruct',
            default_temp=0.4,
            default_max_tokens=700,
        )
    def _coder_llm(self):
        return self._build_llm(
            role='CODER',
            default_model='openrouter/qwen/qwen2.5-coder-14b-instruct',
            default_temp=0.1,
            default_max_tokens=1000,
        )

    # here we instantiate the tools we've previously made
    def _articles_tool(self):
        return ArticlesQueryTool()

    def _codes_tool(self):
        return CodesQueryTool()

   #here we follow the agents 
    @agent
    def manager_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['manager_agent'],
            llm=self._manager_llm(),
            allow_delegation=True,
            tools=[]
        )

    @agent
    def explainer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['explainer_agent'],
            llm=self._explainer_llm(),
            allow_delegation=False,
            cache=True,
            tools=[self._articles_tool()]
        )

    @agent
    def coder_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['coder_agent'],
            llm=self._coder_llm(),
            allow_delegation=False,
            cache=False,
            tools=[self._codes_tool(), FileWriterTool()]
        )

    
    @task
    def routing_task(self) -> Task:
        return Task(config=self.tasks_config['routing_task'])

    @task
    def explanation_task(self) -> Task:
        return Task(config=self.tasks_config['explanation_task'])

    @task
    def coding_task(self) -> Task:
        return Task(
            config=self.tasks_config['coding_task'],
            output_file='output/result.py'
        )

    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.explainer_agent(), self.coder_agent()],
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self.manager_agent(),
            memory=False,
            cache=True,
            verbose=True
        )
     def router_crew(self) -> Crew:
        return Crew(
            agents=[self.manager_agent()],
            tasks=[self.routing_task()],
            process=Process.sequential,
            memory=False,
            cache=True,
            verbose=True,
        )

    def explainer_crew(self) -> Crew:
        return Crew(
            agents=[self.explainer_agent()],
            tasks=[self.explanation_task()],
            process=Process.sequential,
            memory=False,
            cache=True,
            verbose=True,
        )

    def coder_crew(self) -> Crew:
        return Crew(
            agents=[self.coder_agent()],
            tasks=[self.coding_task()],
            process=Process.sequential,
            memory=False,
            cache=True,
            verbose=True,
        )