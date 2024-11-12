from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

import os


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


system_prompt = (
    "You are a assistant agent tasks with handling the user's knowledge bases,"
    " you have tools to do any one of  "
    " Given the following user request, use the appropriate tools to perform any one of the following tasks:\n "
    "add a knowledge base, list existing knowledge bases, specify knowledge bases to use or find a knowledge from their existing knowledge bases. "
    "A knowledge base can be a text that the user inputs, a document or a link to a website"
    "the user can input one of those options, two of those options or all three combined"
    
    
)





prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, call the right tool for the job"
        ),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


model = ChatOpenAI( 
   openai_api_key=os.getenv('OPENAI_API_KEY'),
   model="gpt-3.5-turbo"

)



from langchain_core.tools import tool




@tool
def add_kb(state):
  """
  Add a knowledge base to the user's knowledge bases.
  """
  

  print("kb added")
  return state 



@tool
def specify_kb(state):
  """
  specify which kb to use 
  """
  
  print("kb specified")

  return state








@tool
def list_kb(state):
  """
  List the knowledge bases that the user has added
  """
  print("list of kbs = [kb1, kb2, kb3]")
  return state



@tool
def find_kb(state):
  """
  Find a knowledge base from the user's knowledge bases
  """
  print("KB found !!!")
  return "return KB list"

tools=[add_kb, list_kb,find_kb, specify_kb]


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key="pk-lf-18ba1b8c-64e5-438c-aa82-ec37260b8485",
    secret_key="sk-lf-b4d1320e-2d85-4a5d-b385-0a67e5256d3f",
    host="http://langfu-loadb-keumznrr3fiy-1568287623.us-east-1.elb.amazonaws.com"
)


response = agent_executor.invoke({ "messages": ["I want to add this link to my KB: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/"]}, config={"callbacks": [langfuse_handler]})
print(response)

