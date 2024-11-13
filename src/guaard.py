# from langfuse import Langfuse
# from langfuse.callback import CallbackHandler
# from bs4 import BeautifulSoup
import os
from langchain_core.runnables import RunnableLambda
# from guardrails.hub import SensitiveTopic, ToxicLanguage
from guardrails.hub.guardrails.sensitive_topics.validator import SensitiveTopic
from guardrails.hub.guardrails.toxic_language.validator import ToxicLanguage

# If RestrictToTopic is needed, import it similarly based on its actual location
from guardrails import Guard
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import BaseModel
from typing import Annotated, Literal, List, Dict, Any, TypedDict, Sequence, Optional
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import operator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import os

# api_key = os.environ.get('GUARDRAILS_API_KEY')
# Initialize ChatGPT with the API key from environment variables
llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))


# Supervisor setup
members = ["ToxicityChecker", "SensitiveTopicChecker"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
options = ["FINISH"] + members

class RouteResponse(BaseModel):
    next: Literal[tuple(options)]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

def supervisor_agent(state):
    if state.get("error"):
        return {
            "messages": [HumanMessage(content=state["error"])],
            "next": "FINISH",
            "error": state["error"]
        }

    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    return supervisor_chain.invoke(state)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    error: Optional[str]

def process_user_input(state):
    """Process the user's input to check for sensitive topics and provide a ChatGPT response if valid."""
    
    sensitive_topics = ["politics", "religion", "race", "violence"]
    guard = Guard().use(SensitiveTopic, sensitive_topics=sensitive_topics, on_fail="exception")

    try:
        user_input = state['messages'][-1].content if state['messages'] else ""
        guard.validate(user_input)
        
        response = llm.invoke([HumanMessage(content=user_input)])
        
        chat_response = response.content if hasattr(response, 'content') else "No content available"
        
        return {
            "messages": [HumanMessage(content=f"Sensitive topic check passed. Response: {chat_response}", name="SensitiveTopicChecker")],
            "next": "",
            "error": None
        }
      
    except Exception as e:
        error_message = str(e).split(': ', 1)[-1]  # Extract error message without repetition
        return {
            "messages": [HumanMessage(content=f"Error: Validation failed for field with errors: {error_message}", name="SensitiveTopicChecker")],
            "next": "",
            "error": error_message
        }



def agent_node(state, func, name):
    result = func(state)
    return {
        "messages": result["messages"],
        "next": "",
        "error": result.get("error")
    }

# Rename existing functions to match the new structure
toxicity_node = lambda state: agent_node(state, check_toxicity, "ToxicityChecker")
sensitive_topic_node = lambda state: agent_node(state, process_user_input, "SensitiveTopicChecker")

# Initialize the workflow
workflow = StateGraph(AgentState)
def check_toxicity(state):
    """Check the input for toxic language and return a ChatGPT response if valid."""
    
    guard = Guard().use(ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception")

    try:
        user_input = state['messages'][-1].content if state['messages'] else ""
        guard.validate(user_input)

        response = llm.invoke([HumanMessage(content=user_input)])
        
        chat_response = response.content if hasattr(response, 'content') else "No content available"
        
        return {
            "messages": [HumanMessage(content=f"Toxicity check passed. Response: {chat_response}", name="ToxicityChecker")],
            "next": "",
            "error": None
        }
      
    except Exception as e:
        error_message = str(e).split(': ', 1)[-1]  # Extraire le message d'erreur sans répétition
        return {
            "messages": [HumanMessage(content=f"Error: Validation failed for field with errors: {error_message}", name="ToxicityChecker")],
            "next": "",
            "error": error_message
        }
# Add nodes to the workflow
workflow.add_node("ToxicityChecker", toxicity_node)
workflow.add_node("SensitiveTopicChecker", sensitive_topic_node)
workflow.add_node("supervisor", supervisor_agent)

# Connect edges
for member in members:
    workflow.add_edge(member, "supervisor")

# Add conditional edges from supervisor to workers or END
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END

def get_next_step(x):
    if x.get("error"):
        return "FINISH"
    return x["next"]

workflow.add_conditional_edges("supervisor", get_next_step, conditional_map)

# Set the entry point
workflow.set_entry_point("supervisor")

# Compile the workflow
compiled_workflow = workflow.compile()

def inp(question: str) -> dict:
    return {
        "messages": [HumanMessage(content=question)]
    }

def out(result: dict) -> str:
    if isinstance(result, dict):
        # Check if 'supervisor' is in the result
        if 'supervisor' in result:
            supervisor_data = result['supervisor']
            # Check if there is an error message
            if supervisor_data.get('error'):
                return supervisor_data['error']  # Returns the error message
            if 'messages' in supervisor_data:
                return supervisor_data['messages'][0].content  # Returns the first message
    return str(result)  # Returns the default representation
guardrails_output_parser_chain = RunnableLambda(inp) | compiled_workflow | RunnableLambda(out)



# Export the Runnable instance
all = ["guardrails_output_parser_chain"]
# from langfuse import Langfuse
# from langfuse.callback import CallbackHandler
# config = {
#     "project": {
#         "name": "lg-test",
#         "id": "cm2ky2o9k000li82l00zhksv9"
#     },
#     "org": {
#         "name": "Clouds Cockpit",
#         "id": "cm2kxylfu0001i82lffl0tfto"
#     }
# }
# langfuse_handler = CallbackHandler(
   
#   secret_key="sk-lf-fdb3a37d-943c-48fb-bb0f-b432a3f35cd7",
#    public_key="pk-lf-3cea53a4-785f-4bc3-91a8-d6a61dacc501",
#   host="http://langfuse-alb-service-985109342.us-east-1.elb.amazonaws.com"
# )
compiled_workflow.invoke(
    {
        "messages": [
            
            HumanMessage(content="Donald Trump is one of the most controversial presidents in the history of the United States")
        ]
    })
