
"""Multi-agent supervisor for coordinating research across multiple specialized agents.

This module implements a supervisor pattern where:
1. A supervisor agent coordinates research activities and delegates tasks
2. Multiple researcher agents work on specific sub-topics independently
3. Results are aggregated and compressed for final reporting

The supervisor uses parallel research execution to improve efficiency while
maintaining isolated context windows for each research topic.
"""

import os
import asyncio

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from langgraph.types import Command

from deep_research.prompts import (
    lead_researcher_prompt,
    final_report_generation_prompt
)
from deep_research.research_agent import researcher_agent
from deep_research.state_multi_agent_supervisor import ( 
    ConductResearch, 
    ResearchComplete
)
from deep_research.state_scope import AgentState
from deep_research.utils import get_today_str, think_tool

supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
supervisor_model = init_chat_model(
    model="openai:deepseek-v3-1-terminus",
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

max_researcher_iterations = 6
max_concurrent_researchers = 3

writer_model = init_chat_model(
    model="openai:deepseek-v3-1-terminus",
    max_tokens=32000,
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

async def supervisor(state: AgentState) -> Command[Literal["conduct_parallel_think", "conduct_parallel_research", "final_report_generation"]]:
    """Coordinate research activities.

    Analyzes the research brief and current progress to decide:
    - What research topics need investigation
    - Whether to conduct parallel research
    - When research is complete

    Args:
        state: Current supervisor state with messages and research progress

    Returns:
        Command to proceed to supervisor_tools node with updated state
    """
    supervisor_messages = state.get("supervisor_messages", [])

    system_message = lead_researcher_prompt.format(
        date=get_today_str(), 
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages
    most_recent_message = await supervisor_model_with_tools.ainvoke(messages)

    research_iterations = state.get("research_iterations", 0)
    exceeded_iterations: bool = (research_iterations + 1) >= max_researcher_iterations
    no_tool_calls: bool = not most_recent_message.tool_calls
    research_complete: bool = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    is_conduct_think: bool = any(
        tool_call["name"] == "think_tool" 
        for tool_call in most_recent_message.tool_calls
    )
    is_conduct_research: bool = any(
        tool_call["name"] == "ConductResearch" 
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or no_tool_calls or research_complete:
        return Command(
            goto="final_report_generation",
            update={
                "current_tool": "final_report_generation",
                "supervisor_messages": [most_recent_message],
                "research_iterations": state.get("research_iterations", 0) + 1,
                "notes": get_notes_from_tool_calls(supervisor_messages)
            }
        )
    elif is_conduct_think:
        thinks: list[str] = [
            tool_call["args"]["reflection"] for tool_call in most_recent_message.tool_calls 
            if tool_call["name"] == "think_tool"
        ]
        return Command(
            goto="conduct_parallel_think",
            update={
                "think_contents": "\n\n".join(thinks),
                "current_tool": "conduct_think",
                "supervisor_messages": [most_recent_message],
                "research_iterations": state.get("research_iterations", 0) + 1
            }
        )
    elif is_conduct_research:
        research_topics: list[str] = [
            tool_call["args"]["research_topic"] for tool_call in most_recent_message.tool_calls 
            if tool_call["name"] == "ConductResearch"
        ]
        return Command(
            goto="conduct_parallel_research",
            update={
                "research_topics": "\n\n".join(research_topics),
                "current_tool": "conduct_research",
                "supervisor_messages": [most_recent_message],
                "research_iterations": state.get("research_iterations", 0) + 1
            }
        )
    else:
        return Command(
            goto="final_report_generation",
            update={
                "current_tool": "final_report_generation",
                "supervisor_messages": [most_recent_message],
                "research_iterations": state.get("research_iterations", 0) + 1,
                "notes": get_notes_from_tool_calls(supervisor_messages)
            }
        )

def conduct_parallel_think(state: AgentState) -> Command[Literal["supervisor"]]:
    """Conduct parallel think research.
    
    Args:
        state: Current supervisor state with messages and research progress
        
    Returns:
        Command to proceed to tool_node with updated state
    """
    most_recent_message = state.get("supervisor_messages", [])[-1]
    
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]

    tool_messages: list[ToolMessage] = []
    for tool_call in think_tool_calls:
        observation = think_tool.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
        )

    return Command(
        goto="supervisor",
        update={
            "supervisor_messages": tool_messages,
        }
    )


async def conduct_parallel_research(state: AgentState) -> Command[Literal["supervisor", "final_report_generation"]]:
    """Conduct parallel research.
    
    Args:
        state: Current supervisor state with messages and research progress
        
    Returns:
        Command to proceed to tool_node with updated state
    """
    
    next_step = "supervisor"
    should_end = False
    most_recent_message = state.get("supervisor_messages", [])[-1]
    tool_messages: list[ToolMessage] = []
    all_raw_notes = []

    try:

        conduct_research_calls = [
            tool_call for tool_call in most_recent_message.tool_calls 
            if tool_call["name"] == "ConductResearch"
        ]

        if conduct_research_calls:
            coros = [
                researcher_agent.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }) 
                for tool_call in conduct_research_calls
            ]

            tool_results = await asyncio.gather(*coros)

            research_tool_messages = [
                ToolMessage(
                    content=result.get("compressed_research", "Error synthesizing research report"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ) for result, tool_call in zip(tool_results, conduct_research_calls)
            ]

            tool_messages.extend(research_tool_messages)

            all_raw_notes = [
                "\n".join(result.get("raw_notes", [])) 
                for result in tool_results
            ]

    except Exception as e:
        print(f"Error in supervisor tools: {e}")
        should_end = True
        next_step = "final_report_generation"

    if should_end:
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(state.get("supervisor_messages", []))
            }
        )
    else:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes,
            }
        )

def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history.

    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. When the supervisor delegates research to
    sub-agents via ConductResearch tool calls, each sub-agent returns its
    compressed findings as the content of a ToolMessage. This function
    extracts all such ToolMessage content to compile the final research notes.

    Args:
        messages: List of messages from supervisor's conversation history

    Returns:
        List of research note strings extracted from ToolMessage objects
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

async def final_report_generation(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """

    notes: list[str] = state.get("notes", [])
    findings: str = "\n".join(notes)

    final_report_prompt: str = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    final_report: AIMessage = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }