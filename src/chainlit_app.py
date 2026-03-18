import asyncio
import os
import sys

import chainlit as cl

# Ensure we can import main.py regardless of where Chainlit is started from
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from main import run_research


@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "Welcome to the Research Desk.\n\n"
            "Describe what you want to research, and I'll run the full multi-agent "
            "pipeline (search, filtering, scraping, writing, evaluation) and return "
            "a detailed, well-structured summary.\n\n"
            "You will see both the final answer and the collected sources/notes."
        )
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    task = message.content.strip()
    if not task:
        await cl.Message(content="Please enter a non-empty research query.").send()
        return

    # Inform the user that the pipeline has started
    status_msg = await cl.Message(content="Running research pipeline...").send()

    try:
        # Run the synchronous research pipeline in a thread so we don't block the event loop.
        full_state = await asyncio.to_thread(run_research, task)
    except Exception as e:
        await status_msg.update(content=f"Research pipeline failed: {e}")
        return

    final_answer = full_state.get("final_answer") or full_state.get("answer")
    research_notes = full_state.get("research_notes", [])

    if not final_answer:
        final_answer = "The research pipeline completed, but no final answer was produced."

    # Update status and display structured output
    await status_msg.update(content="Research pipeline completed.")

    # Show final answer
    await cl.Message(content=final_answer).send()

    # Optionally show sources/notes in a collapsible block if we have any
    if research_notes:
        notes_text = "\n\n".join(research_notes)
        await cl.Message(
            content="Collected sources and notes:",
            elements=[
                cl.Text(name="Research Notes", content=notes_text),
            ],
        ).send()

