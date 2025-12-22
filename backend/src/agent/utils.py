import os
from datetime import datetime
from pydantic import (
    BaseModel,
    Field
)
from typing_extensions import Annotated, Literal

from langchain.chat_models import init_chat_model 
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    HumanMessage
)
from langchain_core.tools import InjectedToolArg
from tavily import TavilyClient
from agent.prompts import summarize_webpage_instructions


def get_research_topic(messages: list[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    if len(messages) == 1:
        research_topic: str = messages[-1].content
    else:
        research_topic: str = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic

class Summary(BaseModel):
    """Schema for webpage content summarization."""
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")

def get_current_date() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

summarization_model = init_chat_model(
    model="openai:deepseek-v3-2-251201",
    temperature=0.0,
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)
tavily_client = TavilyClient()

def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    search_results: list[dict] = tavily_search_multiple(
        [query],
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )
    unique_results: dict = deduplicate_search_results(search_results)
    summarized_results: dict[str, dict[str, str]] = process_search_results(unique_results)
    return format_search_output(summarized_results)

def tavily_search_multiple(
    search_queries: list[str], 
    max_results: int = 3, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> list[dict]:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """

    search_docs: list[dict] = []
    for query in search_queries:
        result: dict = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        search_docs.append(result)

    return search_docs

def deduplicate_search_results(search_results: list[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results: dict = {}

    for response in search_results:
        for result in response['results']:
            url: str = result['url']
            if url not in unique_results:
                unique_results[url] = result

    return unique_results

def process_search_results(unique_results: dict) -> dict[str, dict[str, str]]:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """
    summarized_results: dict[str, dict[str, str]] = {}

    for url, result in unique_results.items():
        if not result.get("raw_content"):
            content: str = result['content']
        else:
            content: str = summarize_webpage_content(result['raw_content'])

        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }

    return summarized_results

def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary with key excerpts
    """
    try:
        structured_model = summarization_model.with_structured_output(Summary)
        summary: Summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_instructions.format(
                webpage_content=webpage_content, 
                date=get_current_date()
            ))
        ])

        formatted_summary: str = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

def format_search_output(summarized_results: dict[str, dict[str, str]]) -> str:
    """Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output: str = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output
