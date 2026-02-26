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

    
    def _manager_llm(self):
        return LLM(
            model="groq/llama-3.1-8b-instant", 
            temperature=0.1,
            api_key=os.environ["GROQ_API_KEY"]
        )

    def _explainer_llm(self):
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            temperature=0.5,
            api_key=os.environ["GROQ_API_KEY"]
        )

    def _coder_llm(self):
        return LLM(
            model="groq/deepseek-r1-distill-llama-70b",
            temperature=0.27, #Why 0.27? I really couldn't  decide whether to put 0.2 or 0.3, but I lean towards the later
            api_key=os.environ["GROQ_API_KEY"]
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
            memory=True,
            cache=True,
            verbose=True
        )