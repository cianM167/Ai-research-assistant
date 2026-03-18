import json
import re
import requests
import os
import operator

from typing import TypedDict, List, Optional, Annotated, Dict, Any
import io
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from bs4 import BeautifulSoup
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from pydantic import BaseModel, Field

class ResearchState(TypedDict):
    task: str
    answer: str
    final_answer: str
    critic_feedback: str
    iterations: int
    queries: List[str]
    remaining_urls: List[str]
    current_url: Optional[str]
    current_score: int
    # Each note is a structured record so we can handle citations and attribution.
    research_notes: List[Dict[str, Any]]
    evaluator_score: int
    fix_instructions: str
    raw_search_results: List[dict]
    clear_notes: bool
    data_gap_retries: int

MIN_SOURCES = 3  # minimum number of distinct sources required before writing

load_dotenv()
llm = ChatOllama(model="llama3.2", temperature=0)
search_tool = TavilySearchResults(
    k=10,
    search_depth="advanced",
    include_answer=True
)


class SearchQueries(BaseModel):
    """Structured output for the Architect node."""

    queries: List[str] = Field(description="Up to 3 technical search queries")


class EvaluationFeedback(BaseModel):
    """Structured output for the Evaluator node."""

    score: int = Field(description="Numeric quality score from 1-10")
    fix_instructions: str = Field(
        description="Concrete instructions on how to fix inaccuracies or missing information."
    )


architect_llm = llm.with_structured_output(SearchQueries)
evaluator_llm = llm.with_structured_output(EvaluationFeedback)


def _is_research_retry(state: ResearchState) -> bool:
    """
    Detect whether we are entering a re-search loop due to missing or insufficient data.
    Two triggers:
    - Evaluator has flagged missing information (as in should_continue's 're_search' branch).
    - The router has explicitly requested a re-search due to too few sources (via clear_notes flag).
    """
    # Explicit signal from the router (insufficient sources)
    if state.get("clear_notes"):
        return True

    # Implicit signal from evaluator
    score = state.get("evaluator_score", 10)
    if score < 8:
        notes = state.get("research_notes", [])
        if not notes or "missing" in state.get("fix_instructions", "").lower():
            return True
    return False

def query_generator_node(state: ResearchState):
    print("--- ARCHITECT: GENERATING/REFINING SEARCH QUERIES ---")
    
    # State Cleaner: if we're re-searching due to missing / low-quality data,
    # clear out previous research artifacts so we don't keep junk notes.
    cleaned_state_update = {}
    if _is_research_retry(state):
        print("--- STATE CLEANER: CLEARING PREVIOUS RESEARCH NOTES AND URL STATE ---")
        cleaned_state_update = {
            "research_notes": [],
            "raw_search_results": [],
            "remaining_urls": [],
            "current_url": None,
            "clear_notes": False,
            "data_gap_retries": state.get("data_gap_retries", 0),
        }

    feedback = state.get("fix_instructions", "None")
    context_segment = ""
    if feedback != "None":
        context_segment = f"\nPREVIOUS ATTEMPT FAILED: {feedback}\nAvoid previous mistakes and try a different technical angle."

    prompt = f"""
    You are a Research Architect. Break the user's request into up to 3 precise, technical search queries.
    User Request: {state['task']}{context_segment}
    """

    try:
        structured = architect_llm.invoke(prompt)
        queries = structured.queries or [state["task"]]
    except Exception as e:
        print(f"Architect structured output error: {e}")
        queries = [state["task"]]

    result = {"queries": queries}
    if cleaned_state_update:
        result.update(cleaned_state_update)
    return result

def gather_sources_node(state: ResearchState):
    # 1. Safety check for the 'queries' key
    queries = state.get("queries", [])
    
    # Fallback: If Architect failed, use the original task as the search query
    if not queries:
        queries = [state.get("task", "General research")]

    print(f"--- SEARCH AGENT: SEARCHING FOR {len(queries)} TOPICS (parallel) ---")

    all_results: List[dict] = []

    def _run_search(q: str):
        if not q:
            return []
        try:
            return search_tool.invoke({"query": q}) or []
        except Exception as tool_err:
            print(f"!!! TOOL ERROR for query '{q}': {tool_err} !!!")
            return []

    try:
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
            future_to_query = {executor.submit(_run_search, q): q for q in queries}
            for future in as_completed(future_to_query):
                try:
                    results = future.result()
                    if results and isinstance(results, list):
                        all_results.extend(results)
                except Exception as e:
                    print(f"!!! SEARCH FUTURE ERROR: {e} !!!")
                    continue
    except Exception as e:
        print(f"!!! CRITICAL GATHER ERROR: {e} !!!")

    # 3. ALWAYS return a dictionary. 
    # This prevents the "Node returned None" error.
    return {"raw_search_results": all_results}

def filter_sources_node(state: ResearchState):
    print("--- FILTERING SOURCES: REMOVING JUNK ---")
    raw = state.get("raw_search_results", [])
    if not raw:
        return {"remaining_urls": [], "current_url": None}

    # 1. Hardcoded Heuristics (Instant trash removal)
    blacklist = [
        "facebook.com", "instagram.com", "twitter.com", "x.com", 
        "youtube.com", "youtu.be", "pinterest.com", "quora.com",
        "reddit.com" # Keep reddit only if you want forum opinions
    ]
    
    filtered_items = []
    for item in raw:
        url = item.get("url", "").lower()
        # Skip if blacklisted
        if any(domain in url for domain in blacklist):
            continue
        filtered_items.append(item)

    # 2. LLM-based Ranking
    # We send the snippets to the LLM and ask it to pick the most informative ones
    choices = [{"index": i, "url": item['url'], "snippet": item.get('content', '')[:200]} 
               for i, item in enumerate(filtered_items)]
    
    prompt = f"""
    Task: {state['task']}
    Candidate Sources: {choices}
    
    Evaluate which 3 sources likely contain the most objective, factual, and detailed text 
    relevant to the task. Avoid generic homepages or video links.
    
    Return ONLY a JSON list of the URLs.
    Example: ["https://example.com/article1", "https://site.org/paper"]
    """
    
    try:
        response = llm.invoke(prompt)
        match = re.search(r"\[.*\]", response.content, re.DOTALL)
        if match:
            # Clean up the output to ensure it's a list of strings
            suggested_urls = json.loads(match.group(0))
            # Ensure they are actually in our filtered list (no hallucinations)
            final_urls = [u for u in suggested_urls if any(item['url'] == u for item in filtered_items)]
        else:
            final_urls = [item['url'] for item in filtered_items[:3]]
    except Exception as e:
        print(f"Filter LLM Error: {e}")
        final_urls = [item['url'] for item in filtered_items[:3]]

    # 3. Final Fallback: If LLM rejected everything, just take top 3 non-blacklisted
    if not final_urls:
        final_urls = [item['url'] for item in filtered_items[:3]]

    # Prepare for scraper
    unique_urls = list(dict.fromkeys(final_urls)) # Preserves order unlike set()
    current = unique_urls.pop(0) if unique_urls else None

    return {
        "remaining_urls": unique_urls, 
        "current_url": current
    }

def scrape_source_node(state: ResearchState):
    url = state.get("current_url")
    
    # Safety check: ensure URL is a string and not a dict
    if not url or not isinstance(url, str):
        print(f"!!! INVALID URL SKIP: {url} !!!")
        # Move to next URL even if this one failed
        remaining = state.get("remaining_urls", [])
        next_url = remaining.pop(0) if remaining else None
        return {"current_url": next_url, "remaining_urls": remaining}

    print(f"--- SCRAPER AGENT: READING {url} ---")
    
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()

        # Extract readable text from the page
        soup = BeautifulSoup(res.text, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        page_text = soup.get_text(separator="\n")
        # Trim very long pages to keep within context
        page_text = "\n".join(line.strip() for line in page_text.splitlines() if line.strip())
        page_text = page_text[:8000]

        summary_prompt = (
            "You are a focused research assistant.\n"
            f"URL: {url}\n\n"
            "Extract the main article or body content from the following page text and "
            "summarize it into 3-6 dense paragraphs of factual, objective notes.\n\n"
            "Page text:\n"
            f"{page_text}\n\n"
            "Summary notes:"
        )
        summary = llm.invoke(summary_prompt).content
    except Exception as e:
        summary = f"Scrape failed for {url}: {e}"

    # Accumulate research notes manually so we can reset them on re-search
    existing_notes: List[Dict[str, Any]] = state.get("research_notes", [])
    note_id = len(existing_notes) + 1
    note_record = {
        "id": note_id,
        "url": url,
        "summary": summary,
    }
    updated_notes = existing_notes + [note_record]

    # Update logic
    remaining = state.get("remaining_urls", [])
    next_url = remaining.pop(0) if remaining else None

    # If we've finished all URLs but still have too few sources, mark for at most one re-search
    clear_notes_flag = False
    data_gap_retries = state.get("data_gap_retries", 0)
    if not next_url and not remaining and len(updated_notes) < MIN_SOURCES:
        if data_gap_retries == 0:
            print(f"--- SCRAPER: ONLY {len(updated_notes)} SOURCES FOUND (< {MIN_SOURCES}), FLAGGING ONE RE-SEARCH ---")
            clear_notes_flag = True
            data_gap_retries += 1
        else:
            print(f"--- SCRAPER: ONLY {len(updated_notes)} SOURCES FOUND BUT ALREADY RETRIED ONCE, MOVING ON ---")

    return {
        "research_notes": updated_notes,
        "remaining_urls": remaining,
        "current_url": next_url,
        "clear_notes": clear_notes_flag,
        "data_gap_retries": data_gap_retries,
    }

def writer_node(state: ResearchState):
    print("--- WRITER IS WORKING ---")
    notes: List[Dict[str, Any]] = state.get("research_notes", [])
    # Prepare citations: [Source 1], [Source 2], ...
    sources_for_prompt = [
        f"[Source {n['id']}] {n['url']}\nSummary: {n['summary']}"
        for n in notes
    ]
    sources_block = "\n\n".join(sources_for_prompt) if sources_for_prompt else "No sources collected."

    prompt = (
        "You are a meticulous technical research writer.\n"
        f"Topic: {state['task']}\n\n"
        "You are given structured research notes with explicit source IDs. "
        "Every concrete factual claim in your answer MUST be followed by a citation "
        "in the form [Source 1], [Source 2], etc., matching the IDs from the notes.\n\n"
        f"Research Notes (with sources):\n{sources_block}\n\n"
        f"Feedback from prior attempts (if any): {state.get('critic_feedback', 'None')}\n\n"
        "Write a long, detailed, and well-structured report that:\n"
        "- Is comprehensive and deeply explanatory (aim for at least 800–1200 words when enough information is available).\n"
        "- Organizes content into clear sections with headings and, where useful, bullet lists.\n"
        "- Explains key concepts step by step, with concrete examples where appropriate.\n"
        "- Avoids vague generalities and instead provides specific, grounded details from the notes.\n"
        "- Uses precise, clear language suitable for an experienced developer audience.\n\n"
        "Include a final 'References' section at the end that lists each source as:\n"
        "[Source N] URL\n\n"
        "Final Answer:"
    )
    try:
        response = llm.invoke(prompt)
        content = response.content
    except Exception as e:
        # Graceful degradation if the LLM backend (e.g. Ollama) is unavailable
        content = (
            f"Writer LLM backend error: {e}.\n\n"
            "Please ensure your local LLM server (e.g. Ollama with the 'llama3.2' model) "
            "is running and reachable, then retry this request."
        )
        print(f"Writer error: {e}")

    return {
        "answer": content,
        "final_answer": content,
        "iterations": state.get("iterations", 0) + 1
    }

def evaluator_node(state: ResearchState):
    print("--- EVALUATOR IS WORKING ---")
    
    prompt = f"""
    You are an evaluator. Compare the answer against the research notes and score it.

    Research Notes: {state['research_notes']}
    Answer: {state['final_answer']}

    Evaluate:
    1. Grounding: Are there facts in the answer NOT in the notes?
    2. Accuracy: Does it misinterpret the notes?
    3. Citation quality: Are claims properly linked to plausible sources?

    Return a structured object with:
    - score: integer 1-10
    - fix_instructions: a short description of how to fix any issues.
    """

    # Default values in case structured output fails
    score = 10
    fix_instructions = ""

    try:
        result: EvaluationFeedback = evaluator_llm.invoke(prompt)
        score = result.score
        fix_instructions = result.fix_instructions
    except Exception as e:
        print(f"Evaluator structured output error: {e}. Defaulting to score 10.")

    return {
        "evaluator_score": score,
        "fix_instructions": fix_instructions
    }

def should_continue(state: ResearchState):
    if state.get("iterations", 0) >= 3:
        print("--- MAX RETRIES REACHED ---")
        return "end"

    score = state.get("evaluator_score", 0)
    if score < 8:
        # LOGIC: If we have no notes or the evaluator explicitly mentions 'missing info'
        notes = state.get("research_notes", [])
        if not notes or "missing" in state.get("fix_instructions", "").lower():
            print("--- DATA GAP DETECTED: ROUTING TO RE-SEARCH ---")
            return "re_search"
        
        print("--- WRITING ISSUE: ROUTING TO RE-WRITE ---")
        return "retry_writer"

    return "end"

def should_scrape(state: ResearchState):
    notes = state.get("research_notes", [])

    # If we have a current URL to process, or more in the queue, keep scraping
    if state.get("current_url") or (state.get("remaining_urls") and len(state["remaining_urls"]) > 0):
        print("--- ROUTER: MORE SOURCES FOUND, SCRAPING NEXT ---")
        return "scrape_more"

    # No more URLs to scrape at this point.
    # If we've explicitly flagged that we don't have enough sources and haven't already retried,
    # route once to re-search.
    if state.get("clear_notes") and len(notes) < MIN_SOURCES:
        print(f"--- ROUTER: ONLY {len(notes)} SOURCES COLLECTED (< {MIN_SOURCES}), RE-SEARCHING (ONE-TIME) ---")
        return "re_search"

    # If we are out of URLs but have enough research notes, move to the Writer
    if notes and len(notes) >= MIN_SOURCES:
        print("--- ROUTER: RESEARCH COMPLETE, PROCEEDING TO WRITER ---")
        return "to_writer"

    # Fail-safe: if everything failed and we still have no usable notes, just end
    print("--- ROUTER: NO USABLE SOURCES, ENDING ---")
    return "end"

builder = StateGraph(ResearchState)

builder.add_node("generate_queries", query_generator_node)
builder.add_node("gather_sources", gather_sources_node)
builder.add_node("scrape_sources", scrape_source_node)
builder.add_node("writer", writer_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("filter_sources", filter_sources_node)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "gather_sources")
builder.add_edge("gather_sources", "filter_sources")
builder.add_edge("filter_sources", "scrape_sources")

builder.add_conditional_edges(
    "scrape_sources",
    should_scrape,
    {
        "scrape_more": "scrape_sources",
        "to_writer": "writer",
        "re_search": "generate_queries",
        "end": END,
    },
)

builder.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "re_search": "generate_queries",  # The "Smart" Loop
        "retry_writer": "writer",        # The "Polishing" Loop
        "end": END,
    },
)

graph = builder.compile()


def _run_research_with_logger(task: str, log_stream) -> dict:
    """
    Core implementation that runs the graph while sending all stdout to the given log_stream.
    """
    initial_input = {"task": task}
    full_state: dict = {}

    with contextlib.redirect_stdout(log_stream):
        for output in graph.stream(initial_input):
            for node_name, state_update in output.items():
                print(f"--- {node_name.upper()} COMPLETED ---")

                # Guard rail: only update if state_update is a valid dictionary
                if state_update is not None and isinstance(state_update, dict):
                    full_state.update(state_update)
                else:
                    print(f"WARNING: Node '{node_name}' returned an invalid update: {state_update}")

    return full_state


def run_research(task: str) -> dict:
    """
    Convenience wrapper to run the research graph programmatically.
    Captures the thought process into a string buffer and returns it as debug_log.
    """
    log_buffer = io.StringIO()
    full_state = _run_research_with_logger(task, log_buffer)
    full_state.setdefault("debug_log", log_buffer.getvalue())
    return full_state


def run_research_streaming(task: str, log_stream) -> dict:
    """
    Variant of run_research that writes logs into the provided log_stream.
    The caller can inspect log_stream as the graph runs to stream progress.
    """
    full_state = _run_research_with_logger(task, log_stream)
    # If the caller wants a final snapshot, include the accumulated log contents when available.
    if hasattr(log_stream, "getvalue"):
        full_state.setdefault("debug_log", log_stream.getvalue())
    return full_state


if __name__ == "__main__":
    print("Please enter your input prompt:")
    user_input = input()

    final_state = run_research(user_input)

    if "final_answer" in final_state:
        print("\nPROCESSED ANSWER:")
        print(final_state["final_answer"])