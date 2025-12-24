
"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

import os
from langgraph.graph import StateGraph, START, END

from deep_research.state_scope import AgentState, AgentInputState
from deep_research.research_agent_scope import clarify_with_user, write_research_brief
from deep_research.multi_agent_supervisor import (
    supervisor, 
    conduct_parallel_think,
    conduct_parallel_research,
    final_report_generation
)

from deep_research.state_scope import AgentState

deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

deep_researcher_builder.add_node("supervisor", supervisor)
deep_researcher_builder.add_node("conduct_parallel_think", conduct_parallel_think)
deep_researcher_builder.add_node("conduct_parallel_research", conduct_parallel_research)

deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "supervisor")
deep_researcher_builder.add_edge("final_report_generation", END)

agent = deep_researcher_builder.compile()
