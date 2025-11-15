from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator
import csv
from dotenv import load_dotenv
from movie_db.prompts import return_instructions_root
import json
import requests
from utils.logger import get_logger
import os


_logs = get_logger(__name__)

load_dotenv(".env")
load_dotenv(".secrets")



@tool
def get_history_movie(year:int=2025, genre:str="Drama", limit:int=3):
    """
    Returns upcoming movies from Movies Database API.
    Args:
        year (int): The release year of the movies.
        genre (str): The genre of the movies.
        limit (int): Maximum number of movies to return.
    Returns:
        list[dict]: A list of movie dictionaries
    """
    url = "https://moviesdatabase.p.rapidapi.com/titles/x/upcoming"
    params = {
        'startYear': year,
        'endYear': year,
        'genre': genre,
        'limit': limit
    }
    requests.headers = {'x-rapidapi-host': 'moviesdatabase.p.rapidapi.com', 
                        'x-rapidapi-key': '804de831efmsh6c58705579278d7p11f8eejsnb970e0aed97c'}


    response = requests.get(url, params=params)
    resp_dict = json.loads(response.text)
    movie_list = resp_dict.get("results", [])
    movie_list = "\n".join([f"{i+1}. {fact}\n" for i, fact in enumerate(movie_list)])
    return movie_list

@tool()
def fetch_on_air_movie(max_results=100):
    """Load a list of currently on-air movies from a local CSV.

    Args:
        max_results (int): Maximum number of movies to return.

    Returns:
        list[dict]: A list of movie dictionaries with keys such as
            'title', 'year', 'genre', 'director', and 'plot'. Returns an
            empty list if the CSV cannot be found or on parse errors.
    """
    import os
    on_air_movies = []
    try:
        # Prefer the CSV we added for the assignment; keep path relative to this file
        csv_path = os.path.join(os.path.dirname(__file__), 'on_air_movie.csv')
        with open(csv_path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if i >= max_results:
                    break
                # normalize keys and types
                movie = {k.strip(): v.strip() for k, v in row.items() if v is not None}
                on_air_movies.append(movie)
    except FileNotFoundError:
        # If CSV is missing, return an empty list (caller can handle)
        return []
    except Exception:
        # For any other parsing error, return empty list to match existing behavior
        return []

    return on_air_movies

def get_model_with_tools():
    model = init_chat_model(
        "openai:gpt-4o-mini",
        temperature=0.7
    )
    # Augment the LLM with tools
    tools = [get_history_movie, fetch_on_air_movie]
    model_with_tools = model.bind_tools(tools)
    return model_with_tools

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    model_with_tools = get_model_with_tools()
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with checking on air movies and historic movies"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: dict):
    """Performs the tool call"""
    tools = [get_history_movie, fetch_on_air_movie]
    tools_by_name = {tool.name: tool for tool in tools}

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def get_movies_chat_agent():
    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")
    return agent_builder.compile()