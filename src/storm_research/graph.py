"""STORM Research Assistant Main Graph Definition

This module defines the LangGraph graph that orchestrates the research process.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Literal, cast
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# from langgraph.checkpoint.memory import InMemorySaver  # LangGraph API handles this automatically
from langgraph.constants import Send

from storm_research.state import (
    InterviewState,
    InputState,
    OutputState,
    ResearchGraphState,
    Analyst,
    Perspectives,
    SearchQuery,
)
from storm_research.prompts import (
    ANALYST_INSTRUCTIONS,
    QUESTION_INSTRUCTIONS,
    ANSWER_INSTRUCTIONS,
    SEARCH_INSTRUCTIONS,
    SECTION_WRITER_INSTRUCTIONS,
    REPORT_WRITER_INSTRUCTIONS,
    INTRO_CONCLUSION_INSTRUCTIONS,
)
from storm_research.configuration import Configuration
from storm_research.tools import get_search_tools
from storm_research.utils import load_chat_model, generate_thread_id


# ====================== Analyst Generation Node ======================


async def create_analysts(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Generate analyst personas tailored to the research topic

    Each analyst contributes to the research with unique perspectives and expertise.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    topic = state["messages"][-1].content
    max_analysts = state.get("max_analysts", configuration.max_analysts)

    # Configure model for structured output
    structured_model = model.with_structured_output(Perspectives)

    # Construct prompt
    system_message = ANALYST_INSTRUCTIONS.format(
        topic=topic,
        human_analyst_feedback="",  # User feedback is empty
        max_analysts=max_analysts,
    )

    # Generate analysts
    result = await structured_model.ainvoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the set of analysts."),
        ]
    )

    return {"analysts": result.analysts, "topic": topic}


# ====================== Interview Nodes ======================


async def generate_question(state: InterviewState, config: RunnableConfig) -> dict:
    """Generate questions from analyst to expert

    Creates insightful questions based on the analyst's persona
    and previous conversation content.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    analyst = state["analyst"]
    messages = state["messages"]

    # Construct question generation prompt
    system_message = QUESTION_INSTRUCTIONS.format(goals=analyst.persona)

    # Generate question
    question = await model.ainvoke([SystemMessage(content=system_message)] + messages)

    return {"messages": [question]}


async def search_web(state: InterviewState, config: RunnableConfig) -> dict:
    """Search for relevant information on the web

    Analyzes conversation content to generate appropriate search queries
    and searches for relevant information on the web.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    search_tools = get_search_tools(config)

    # Generate search query
    structured_model = model.with_structured_output(SearchQuery)
    search_query = await structured_model.ainvoke(
        [SystemMessage(content=SEARCH_INSTRUCTIONS)] + state["messages"]
    )

    # Perform web search
    search_results = await search_tools.search_web(search_query.search_query)

    return {"context": [search_results]}


async def search_arxiv(state: InterviewState, config: RunnableConfig) -> dict:
    """Search for academic papers on ArXiv

    Analyzes conversation content to generate appropriate search queries
    and searches for relevant papers on ArXiv.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    search_tools = get_search_tools(config)

    # Generate search query
    structured_model = model.with_structured_output(SearchQuery)
    search_query = await structured_model.ainvoke(
        [SystemMessage(content=SEARCH_INSTRUCTIONS)] + state["messages"]
    )

    # Perform ArXiv search
    search_results = await search_tools.search_arxiv(search_query.search_query)

    return {"context": [search_results]}


async def generate_answer(state: InterviewState, config: RunnableConfig) -> dict:
    """Generate answer to questions as an expert

    Creates detailed and accurate answers from an expert perspective
    based on the searched context.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Construct answer generation prompt
    system_message = ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)

    # Generate answer
    answer = await model.ainvoke([SystemMessage(content=system_message)] + messages)

    # Mark as expert answer
    answer.name = "expert"

    return {"messages": [answer]}


async def save_interview(state: InterviewState) -> dict:
    """Save completed interview content

    Converts conversation content to string format and saves it.
    """
    messages = state["messages"]
    interview_content = get_buffer_string(messages)

    return {"interview": interview_content}


def route_messages(
    state: InterviewState, name: str = "expert"
) -> Literal["ask_question", "save_interview"]:
    """Determine next step based on interview progress

    Saves when maximum turns are reached or interview is complete,
    otherwise generates additional questions.
    """
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 3)

    # Check number of expert responses
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # Check if maximum turns reached
    if num_responses >= max_num_turns:
        return "save_interview"

    # Check for interview end signal
    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        return "save_interview"

    return "ask_question"


async def write_section(state: InterviewState, config: RunnableConfig) -> dict:
    """Write report section based on interview content

    Organizes interview content from the analyst's perspective
    to write a section of the report.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    context = state["context"]
    analyst = state["analyst"]

    # Construct section writing prompt
    system_message = SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description)

    # Write section
    section = await model.ainvoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Use this source to write your section: {context}"),
        ]
    )

    return {"sections": [section.content]}


# ====================== Report Writing Nodes ======================


def initiate_all_interviews(state: ResearchGraphState) -> List[Send]:
    """Start interviews for all analysts simultaneously

    Initiates independent interview processes for each analyst.
    """
    topic = state.get("topic", "")

    # Start interview for each analyst
    return [
        Send(
            "conduct_interview",
            {
                "analyst": analyst,
                "messages": [
                    HumanMessage(
                        content=f"So you said you were writing an article on {topic}?"
                    )
                ],
                "max_num_turns": state.get("max_num_turns", 3),
            },
        )
        for analyst in state["analysts"]
    ]


async def write_report(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Write report body by integrating all sections

    Synthesizes sections written by each analyst
    into a coherent integrated report.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    sections = state["sections"]
    topic = state.get("topic", "")

    # Connect all sections
    formatted_sections = "\n\n".join(sections)

    # Construct report writing prompt
    system_message = REPORT_WRITER_INSTRUCTIONS.format(
        topic=topic, context=formatted_sections
    )

    # Write report
    report = await model.ainvoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="Write a report based upon these memos."),
        ]
    )

    return {"content": report.content}


async def write_introduction(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Write report introduction

    Creates an engaging introduction that summarizes
    the entire research and captures reader interest.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    sections = state["sections"]
    topic = state.get("topic", "")

    # Connect all sections
    formatted_sections = "\n\n".join(sections)

    # Construct introduction writing prompt
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic, formatted_str_sections=formatted_sections
    )

    # Write introduction
    intro = await model.ainvoke(
        [instructions, HumanMessage(content="Write the report introduction")]
    )

    return {"introduction": intro.content}


async def write_conclusion(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Write report conclusion

    Creates a conclusion that summarizes key findings
    and suggests future research directions.
    """
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    sections = state["sections"]
    topic = state.get("topic", "")

    # Connect all sections
    formatted_sections = "\n\n".join(sections)

    # Construct conclusion writing prompt
    instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
        topic=topic, formatted_str_sections=formatted_sections
    )

    # Write conclusion
    conclusion = await model.ainvoke(
        [instructions, HumanMessage(content="Write the report conclusion")]
    )

    return {"conclusion": conclusion.content}


async def finalize_report(state: ResearchGraphState) -> dict:
    """Assemble final report

    Combines introduction, body, and conclusion
    to create the completed report.
    """
    content = state["content"]

    # Handle case where content arrives as a list
    if isinstance(content, list):
        content = "\n\n".join(str(c) for c in content)

    # Remove "## Insights" title
    if content.startswith("## Insights"):
        content = content.strip("## Insights")

    # Separate Sources section
    sources = None
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None

    # Assemble final report
    final_report = (
        state["introduction"]
        + "\n\n---\n\n## Main Idea\n\n"
        + content
        + "\n\n---\n\n"
        + state["conclusion"]
    )

    # Add Sources section
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources

    return {
        "final_report": final_report,
        "messages": [HumanMessage(content=final_report)],
    }


# ====================== Obsidian Integration Node ======================


def _sanitize_filename(name: str) -> str:
    """Remove characters that are invalid in file names."""
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()


async def save_to_obsidian(state: ResearchGraphState, config: RunnableConfig) -> dict:
    """Save the final report as a Markdown file in the Obsidian vault.

    Adds YAML frontmatter (tags, date, topic) and writes the file to
    the configured Research folder inside the Obsidian vault.
    """
    configuration = Configuration.from_runnable_config(config)
    vault_path = Path(configuration.obsidian_vault_path)
    folder = configuration.obsidian_folder

    # Build target directory
    target_dir = vault_path / folder
    target_dir.mkdir(parents=True, exist_ok=True)

    # Extract topic – prefer explicit topic field, fall back to first user message
    topic = state.get("topic") or ""
    if not topic and state.get("messages"):
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage) and msg.content:
                topic = msg.content
                break
    topic = topic or "Research"

    # Build filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    safe_topic = _sanitize_filename(topic)
    filename = f"{safe_topic} - STORM Research ({date_str}).md"
    file_path = target_dir / filename

    # Build YAML frontmatter
    frontmatter = (
        "---\n"
        f"title: \"{safe_topic} - STORM Research\"\n"
        f"date: {date_str}\n"
        f"topic: \"{safe_topic}\"\n"
        "tags:\n"
        "  - storm-research\n"
        "  - auto-generated\n"
        "---\n\n"
    )

    # Generate Korean summary using LLM
    model = load_chat_model(configuration.model)
    summary = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful translator and summarizer. "
                    "Given an English research report, produce a concise Korean summary. "
                    "Format:\n"
                    "1. 한 줄 한국어 제목 (# 헤딩)\n"
                    "2. '## 핵심 요약' 섹션: 리포트의 핵심 내용을 3~5개 bullet point로 요약\n"
                    "3. '## 주요 키워드' 섹션: 주요 키워드 5~8개를 나열\n"
                    "Output ONLY the Korean summary in Markdown. No English."
                )
            ),
            HumanMessage(content=state["final_report"]),
        ]
    )
    korean_section = summary.content + "\n\n---\n\n"

    # Write file
    file_path.write_text(
        frontmatter + korean_section + state["final_report"], encoding="utf-8"
    )

    return {"file_path": str(file_path)}


# ====================== Graph Build Functions ======================


def build_interview_graph():
    """Create interview subgraph

    Creates a subgraph that manages the interview process
    for a single analyst.
    """
    builder = StateGraph(InterviewState)

    # Add nodes
    builder.add_node("ask_question", generate_question)
    builder.add_node("search_web", search_web)
    builder.add_node("search_arxiv", search_arxiv)
    builder.add_node("answer_question", generate_answer)
    builder.add_node("save_interview", save_interview)
    builder.add_node("write_section", write_section)

    # Define edges
    builder.add_edge(START, "ask_question")
    builder.add_edge("ask_question", "search_web")
    builder.add_edge("ask_question", "search_arxiv")
    builder.add_edge("search_web", "answer_question")
    builder.add_edge("search_arxiv", "answer_question")
    builder.add_conditional_edges(
        "answer_question", route_messages, ["ask_question", "save_interview"]
    )
    builder.add_edge("save_interview", "write_section")
    builder.add_edge("write_section", END)

    # LangGraph API automatically manages checkpointer
    interview_graph = builder.compile().with_config(run_name="Conduct Interview")

    return interview_graph


def build_research_graph():
    """Create main research graph

    Creates the main graph that orchestrates
    the entire research process.
    """
    # Create interview subgraph
    interview_graph = build_interview_graph()

    # Main graph builder - specify input/output schemas
    builder = StateGraph(ResearchGraphState, input=InputState, output=OutputState)

    # Add nodes
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("conduct_interview", interview_graph)
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)
    builder.add_node("save_to_obsidian", save_to_obsidian)

    # Define edges
    builder.add_edge(START, "create_analysts")
    builder.add_conditional_edges(
        "create_analysts", initiate_all_interviews, ["conduct_interview"]
    )

    # Report writing phase
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")

    # Generate final report
    builder.add_edge(
        ["write_conclusion", "write_report", "write_introduction"], "finalize_report"
    )
    builder.add_edge("finalize_report", "save_to_obsidian")
    builder.add_edge("save_to_obsidian", END)

    # LangGraph API automatically manages checkpointer
    graph = builder.compile()

    return graph


# ====================== Main Graph Instance ======================

# Graph instance for use in LangGraph Studio
graph = build_research_graph()
