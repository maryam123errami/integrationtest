import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from typing import List
from typing_extensions import TypedDict


from langfuse import Langfuse
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    
    public_key="pk-lf-18ba1b8c-64e5-438c-aa82-ec37260b8485",
    secret_key="sk-lf-b4d1320e-2d85-4a5d-b385-0a67e5256d3f",
    host="http://langfu-loadb-keumznrr3fiy-1568287623.us-east-1.elb.amazonaws.com"
)

class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

prompt = """
    For the following task, make plans that can solve the problem step by step. For each plan, indicate \
    which external tool together with tool input to retrieve evidence. You can store the evidence into a \
    variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

    Tools can be one of the following:
    (1) emailSender[input]: Worker that save the email data in a json file. Useful when you need to save the email data (the receiver, subject and content ).
    (2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
    world knowledge and common sense. Prioritize it when you are confident in solving the problem
    yourself. Input can be any instruction.

    For example,
    Task: send an email to saad@gmail.com to remind him of our meeting tomorrow at the office at 10 am
    Plan: Given the receiver the contexte reminder of our meeting tomorrow at the office at 10 am, drafta profissional concise email. #E1 = LLM[draft an email to remind saad@gmail.com of our meeting tommorrow at 10 am]
    Plan: save the email data in a json file. #E2 = emailSender[#E1]


    Begin!
    Describe your plans with rich details. Each Plan should be followed by only one #E.

    Task: {task}
"""

# model.invoke(prompt.format(task="send an email to saad@gmail.com to remind him of our meeting tomorrow"))

import re

from langchain_core.prompts import ChatPromptTemplate

def extract_email(task:str) -> str:
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, task)
    return match.group(0) if match else ""


def format_email_content(input_string):
    # Extract the content part
    content_match = re.search(r"content='(.*?)'", input_string, re.DOTALL)
    if not content_match:
        return None, None

    full_content = content_match.group(1)

    # Extract subject
    subject_match = re.search(r"Subject: (.*?)\\n", full_content)
    subject = subject_match.group(1) if subject_match else ""

    # Remove subject line from content
    content = re.sub(r"Subject: .*?\\n\\n", "", full_content, flags=re.DOTALL)

    # Clean up newlines and extra spaces
    content = re.sub(r"\\n", " ", content)
    content = re.sub(r"\s+", " ", content).strip()

    return subject, content
# Regex to match expressions of the form E#... = ...[...]
regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
planner = prompt_template | model


def get_plan(state: ReWOO):
    task = state["task"]
    result = planner.invoke({"task": task})
    # Find all matches in the sample text
    matches = re.findall(regex_pattern, result.content)
    return {"steps": matches, "plan_string": result.content}

from langchain.agents import tool
import json
@tool
def sendEmail(email_data: str) -> str:
    """Tool useful to store the email data in a json file"""
    try:
        # Parse the email_data string into a dictionary
        email_dict = json.loads(email_data)
        subject, content =format_email_content(email_dict['content'])
        email_dict['subject'] = subject
        email_dict['content']= content
        
        # Ensure all required fields are present
        required_fields = ["receiver","subject", "content"]
        if not all(field in email_dict for field in required_fields):
            raise ValueError("Missing required fields in email data")

        # Write the email data to a JSON file
        with open("email_data.json", "w") as f:
            json.dump(email_dict, f, indent=2)
        
        return f"Email data saved successfully for {email_dict['receiver']}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON format in email data"
    except Exception as e:
        return f"Error saving email data: {str(e)}"

def _get_current_task(state: ReWOO):
    if "results" not in state or state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


def tool_execution(state: ReWOO):
    """Worker node that executes the tools of a given plan."""
    _step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    _results = (state["results"] or {}) if "results" in state else {}
    
    for k, v in _results.items():
        tool_input = tool_input.replace(k, v)
    
    if tool == "emailSender":
        # Extract email from the original task
        receiver_email = extract_email(state["task"])
        
        # Prepare email data
        email_data = {
            "receiver": receiver_email,
            "content": tool_input.split('additional_kwargs')[0]
        }
        
        # Convert email_data to JSON string
        email_json = json.dumps(email_data)
        
        result = sendEmail.invoke(email_json)
    elif tool == "LLM":
        result = model.invoke(tool_input)
    else:
        raise ValueError(f"Unknown tool: {tool}")
    
    _results[step_name] = str(result)
    return {"results": _results}


solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""


def solve(state: ReWOO):
    plan = ""
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = (state["results"] or {}) if "results" in state else {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
    prompt = solve_prompt.format(plan=plan, task=state["task"])
    result = model.invoke(prompt)
    return {"result": result.content}

def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"

from langgraph.graph import END, StateGraph, START

graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge("plan", "tool")
graph.add_edge("solve", END)
graph.add_conditional_edges("tool", _route)
graph.add_edge(START, "plan")

graph = graph.compile()

task = "send an email to saad@gmail.com to let him know that project is finished"
resp=graph.invoke({"task": task})
print(resp)



# # for s in app.stream({"task": task}):
# #     print(s)
# #     print("---")



# from langchain_core.runnables import RunnableLambda


# def inp(task: str) -> dict:
#         return {"task": task}

# def out(state: ReWOO) -> str:
#         if "result" in state:
#             return state["result"]
#         else:
#             return str(state)

# chain = RunnableLambda(inp) | graph | RunnableLambda(out)

# task = "send an email to saad@gmail.com to let him know that project is finished"
# chain.invoke({"task":task}, config={"callbacks": [langfuse_handler]})
