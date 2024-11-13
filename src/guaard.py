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
from openai import AsyncOpenAI
import os

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Initialize environment variables
api_key = os.environ.get('GUARDRAILS_API_KEY')


# Define workflow members and system prompt
members = ["ToxicityChecker", "SensitiveTopicChecker"]
system_prompt = """
You are a supervisor tasked with managing a conversation between the following workers: {members}. 
Given the following user request, respond with the worker to act next. 
Each worker will perform a task and respond with their results and status. 
When finished, respond with FINISH.
"""
options = ["FINISH"] + members

class RouteResponse(BaseModel):
    next: Literal[tuple(options)]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    error: Optional[str]

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), members=", ".join(members))

def check_toxicity(state: AgentState) -> AgentState:
    """Check the input for toxic language."""
    guard = Guard().use(ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception")

    try:
        user_input = state['messages'][-1].content if state['messages'] else ""
        guard.validate(user_input)
        
        response = llm.invoke([HumanMessage(content=user_input)])
        chat_response = getattr(response, 'content', "No content available")
        
        return {
            "messages": [HumanMessage(content=f"Toxicity check passed. Response: {chat_response}", 
                                    name="ToxicityChecker")],
            "next": "",
            "error": None
        }
    except Exception as e:
        error_message = str(e).split(': ', 1)[-1]
        return {
            "messages": [HumanMessage(content=f"Error: {error_message}", 
                                    name="ToxicityChecker")],
            "next": "",
            "error": error_message
        }

def process_user_input(state: AgentState) -> AgentState:
    """Process user input for sensitive topics."""
    sensitive_topics = ["politics", "religion", "race", "violence"]
    guard = Guard().use(SensitiveTopic, sensitive_topics=sensitive_topics, on_fail="exception")

    try:
        user_input = state['messages'][-1].content if state['messages'] else ""
        guard.validate(user_input)
        
        response = llm.invoke([HumanMessage(content=user_input)])
        chat_response = getattr(response, 'content', "No content available")
        
        return {
            "messages": [HumanMessage(content=f"Sensitive topic check passed. Response: {chat_response}", 
                                    name="SensitiveTopicChecker")],
            "next": "",
            "error": None
        }
    except Exception as e:
        error_message = str(e).split(': ', 1)[-1]
        return {
            "messages": [HumanMessage(content=f"Error: {error_message}", 
                                    name="SensitiveTopicChecker")],
            "next": "",
            "error": error_message
        }

def supervisor_agent(state: AgentState) -> AgentState:
    """Supervise the workflow and determine next steps."""
    if state.get("error"):
        return {
            "messages": [HumanMessage(content=state["error"])],
            "next": "FINISH",
            "error": state["error"]
        }

    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    return supervisor_chain.invoke(state)

def agent_node(state: AgentState, func, name: str) -> AgentState:
    """Generic agent node function."""
    result = func(state)
    return {
        "messages": result["messages"],
        "next": "",
        "error": result.get("error")
    }

# Create node functions
toxicity_node = lambda state: agent_node(state, check_toxicity, "ToxicityChecker")
sensitive_topic_node = lambda state: agent_node(state, process_user_input, "SensitiveTopicChecker")

# Initialize and configure workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("ToxicityChecker", toxicity_node)
workflow.add_node("SensitiveTopicChecker", sensitive_topic_node)
workflow.add_node("supervisor", supervisor_agent)

# Add edges
for member in members:
    workflow.add_edge(member, "supervisor")

# Configure conditional routing
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END

def get_next_step(x: AgentState) -> str:
    return "FINISH" if x.get("error") else x["next"]

workflow.add_conditional_edges("supervisor", get_next_step, conditional_map)
workflow.set_entry_point("supervisor")

# Compile workflow
compiled_workflow = workflow.compile()

# Input/Output handlers
def inp(question: str) -> AgentState:
    return {"messages": [HumanMessage(content=question)]}

def out(result: dict) -> str:
    if isinstance(result, dict) and 'supervisor' in result:
        supervisor_data = result['supervisor']
        if supervisor_data.get('error'):
            return supervisor_data['error']
        if 'messages' in supervisor_data:
            return supervisor_data['messages'][0].content
    return str(result)

# Create the final chain
guardrails_output_parser_chain = RunnableLambda(inp) | compiled_workflow | RunnableLambda(out)

# Example usage
if __name__ == "__main__":
    # Test the workflow with a sample input
    result = compiled_workflow.invoke({"messages": [HumanMessage(content="dirty")]})
    response = result["messages"][-1]
    print(response)
    print(out(result))