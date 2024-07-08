from langchain import hub
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_react_agent
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools import ShellTool
from dotenv import load_dotenv

load_dotenv()

search = GoogleSerperAPIWrapper()
shell_tool = ShellTool(ask_human_input=False)

tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to find information on the internet",
    ),
    Tool(
        name="Shell",
        func=shell_tool.run,
        description="Useful for when you need to run shell commands",
    )
]

model = ChatOpenAI(model="gpt-4o")
prompt = hub.pull("aws")


agent = create_react_agent(ChatOpenAI(temperature=0), tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

while True:
    query = input("You: ")
    if query == "exit":
        break
    result = agent_executor.invoke({"input": query})