import os
from apikey import apikey, tavily_key
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import streamlit as st

# set the API-keys for OpenAI and Tavily
os.environ['OPENAI_API_KEY'] = apikey
os.environ['TAVILY_API_KEY'] = tavily_key

# create GPT-3.5 instance
llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.7, max_tokens=4096)

# use Tavily as the tool for websearching
tools = [TavilySearchResults(max_results=1)]

# following the Langchain documentation
prompt = hub.pull("hwchase17/openai-tools-agent")

# construct the OpenAI tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# create the agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# create webapp
st.title('GPT Websearch Agent🧬⚗️')
input = st.text_input('What do you want to know?')

# show response if a prompt is entered
if input:
    response=agent_executor.invoke({"input": {input}})

    st.write(response['output'])