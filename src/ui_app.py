import asyncio
import html
import os
import sys
import uuid
import io
from typing import Dict, Any

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse

# Ensure we can import main.py regardless of where uvicorn is started from
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from main import run_research_streaming


app = FastAPI(title="Local Research Desk")


try:
    import markdown  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    markdown = None


# In-memory job store for simple streaming via polling
JOBS: Dict[str, Dict[str, Any]] = {}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Local Research Desk</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #0f172a;
      color: #e5e7eb;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      min-height: 100vh;
    }
    .container {
      max-width: 960px;
      width: 100%;
      padding: 2rem 1.5rem 3rem;
    }
    h1 {
      font-size: 1.8rem;
      margin-bottom: 0.5rem;
      color: #f9fafb;
    }
    p.subtitle {
      margin-top: 0;
      margin-bottom: 1.5rem;
      color: #9ca3af;
    }
    form {
      margin-bottom: 2rem;
      background: #020617;
      border-radius: 0.75rem;
      padding: 1.25rem 1.5rem 1.5rem;
      box-shadow: 0 18px 45px rgba(15, 23, 42, 0.8);
      border: 1px solid #1f2937;
    }
    label {
      display: block;
      font-size: 0.95rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: #e5e7eb;
    }
    textarea {
      width: 100%;
      min-height: 120px;
      border-radius: 0.5rem;
      border: 1px solid #374151;
      background: #020617;
      color: #e5e7eb;
      padding: 0.75rem 0.9rem;
      font-size: 0.95rem;
      resize: vertical;
      outline: none;
      box-sizing: border-box;
    }
    textarea:focus {
      border-color: #3b82f6;
      box-shadow: 0 0 0 1px #3b82f6;
    }
    button {
      margin-top: 0.9rem;
      padding: 0.55rem 1.2rem;
      border-radius: 999px;
      border: none;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      background: linear-gradient(to right, #3b82f6, #22c55e);
      color: #0b1120;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }
    button:hover {
      filter: brightness(1.1);
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.25rem 0.7rem;
      border-radius: 999px;
      background: rgba(31, 41, 55, 0.9);
      color: #9ca3af;
      font-size: 0.8rem;
      margin-bottom: 1.25rem;
    }
    .result-card {
      background: #020617;
      border-radius: 0.75rem;
      padding: 1.5rem 1.5rem 1.2rem;
      border: 1px solid #111827;
      box-shadow: 0 18px 45px rgba(15, 23, 42, 0.8);
    }
    .result-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin-top: 0;
      margin-bottom: 0.75rem;
      color: #f9fafb;
    }
    .result-section-title {
      font-size: 0.9rem;
      font-weight: 600;
      margin-top: 1.25rem;
      margin-bottom: 0.35rem;
      color: #9ca3af;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .answer-body {
      white-space: normal;
      word-wrap: break-word;
      margin: 0;
      font-size: 0.92rem;
      line-height: 1.6;
      color: #e5e7eb;
    }
    .answer-body h1, .answer-body h2, .answer-body h3 {
      margin-top: 1.1rem;
      margin-bottom: 0.4rem;
      color: #f9fafb;
    }
    .answer-body code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      background: #020617;
      padding: 0.1rem 0.3rem;
      border-radius: 0.25rem;
      font-size: 0.85em;
    }
    .answer-body pre code {
      display: block;
      padding: 0.7rem 0.8rem;
      overflow-x: auto;
    }
    .answer-body ul, .answer-body ol {
      padding-left: 1.2rem;
    }
    .answer-body p {
      margin-top: 0.35rem;
      margin-bottom: 0.35rem;
    }
    .notes {
      margin-top: 0.5rem;
      padding: 0.85rem 0.9rem;
      border-radius: 0.60rem;
      background: #020617;
      border: 1px dashed #1f2937;
      max-height: 260px;
      overflow-y: auto;
      font-size: 0.85rem;
    }
    .notes pre {
      color: #9ca3af;
    }
    .log {
      margin-top: 0.5rem;
      padding: 0.85rem 0.9rem;
      border-radius: 0.60rem;
      background: #020617;
      border: 1px dashed #1f2937;
      max-height: 220px;
      overflow-y: auto;
      font-size: 0.8rem;
      color: #9ca3af;
    }
    .feedback {
      margin-top: 0.4rem;
      padding: 0.55rem 0.7rem;
      border-radius: 0.5rem;
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid #1f2937;
      font-size: 0.8rem;
      color: #e5e7eb;
    }
  </style>
  <script>
    document.addEventListener("DOMContentLoaded", function () {
      const form = document.querySelector("form");
      if (!form) return;
      const button = form.querySelector("button[type='submit']");
      const status = document.getElementById("status-text");
      const answerHtmlEl = document.getElementById("answer-html");
      const notesEl = document.getElementById("notes");
      const logEl = document.getElementById("log");
      const feedbackEl = document.getElementById("feedback");
      const resultCard = document.getElementById("result-card");

      async function pollStatus(jobId) {
        try {
          const res = await fetch("/status?job_id=" + encodeURIComponent(jobId));
          if (!res.ok) {
            throw new Error("Status request failed");
          }
          const data = await res.json();

          if (status) {
            status.textContent = "Status: " + data.status;
          }
          if (answerHtmlEl) {
            answerHtmlEl.innerHTML = data.answer_html || "";
          }
          if (notesEl) {
            notesEl.textContent = (data.notes || []).join("\\n\\n");
          }
          if (logEl) {
            logEl.textContent = data.debug_log || "";
          }
          if (feedbackEl) {
            const fb = data.agent_feedback || {};
            const score = fb.evaluator_score;
            const fix = fb.fix_instructions || "";
            let text = "";
            if (score !== undefined && score !== null) {
              text += "Evaluator score: " + score + "/10";
            }
            if (fix) {
              if (text) text += "\\n";
              text += "Fix instructions: " + fix;
            }
            feedbackEl.textContent = text;
          }

          if (resultCard && (data.answer || data.answer_html || data.debug_log)) {
            resultCard.style.display = "block";
          }

          if (data.status === "running" || data.status === "pending") {
            // Continue polling
            setTimeout(function () {
              pollStatus(jobId);
            }, 1500);
          } else {
            // Completed or error – re-enable button
            if (button) {
              button.disabled = false;
              button.textContent = "Run research";
            }
          }
        } catch (e) {
          console.error(e);
          if (status) {
            status.textContent = "Error while polling job status.";
          }
          if (button) {
            button.disabled = false;
            button.textContent = "Run research";
          }
        }
      }

      form.addEventListener("submit", async function (event) {
        event.preventDefault();
        const textarea = document.getElementById("prompt");
        const prompt = textarea ? textarea.value : "";
        if (!prompt.trim()) {
          if (status) {
            status.textContent = "Please provide a non-empty prompt.";
          }
          return;
        }

        if (button) {
          button.disabled = true;
          button.textContent = "Running research...";
        }
        if (status) {
          status.textContent = "Starting research job...";
        }

        try {
          const res = await fetch("/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt }),
          });
          if (!res.ok) {
            throw new Error("Failed to start job");
          }
          const data = await res.json();
          if (!data.job_id) {
            throw new Error("No job_id returned");
          }
          pollStatus(data.job_id);
        } catch (e) {
          console.error(e);
          if (status) {
            status.textContent = "Failed to start research job.";
          }
          if (button) {
            button.disabled = false;
            button.textContent = "Run research";
          }
        }
      });
    });
  </script>
</head>
<body>
  <div class="container">
    <div class="pill">
      <span>🧪 Local Research Desk</span>
      <span>•</span>
      <span>LangGraph + Tavily + Ollama</span>
    </div>
    <h1>Research Assistant</h1>
    <p class="subtitle">
      Ask a deep technical or research question. The agent will search, filter sources,
      scrape content, and write a detailed, grounded summary.
    </p>

    <form method="post" action="/run">
      <label for="prompt">Research prompt</label>
      <textarea id="prompt" name="prompt" placeholder="e.g. Explain how LangGraph can be used to build a multi-agent research assistant.">__PROMPT__</textarea>
      <button type="submit">Run research</button>
    </form>
    <p id="status-text" class="subtitle"></p>

    <div class="result-card" id="result-card" style="display: none;">
      <h2 class="result-title">Answer</h2>
      <div class="answer-body" id="answer-html"></div>
      <div class="result-section">
        <div class="result-section-title">Sources &amp; notes</div>
        <div class="notes">
          <pre id="notes"></pre>
        </div>
        <div class="result-section-title">Agent feedback</div>
        <div class="feedback">
          <pre id="feedback"></pre>
        </div>
        <div class="result-section-title">Thought process</div>
        <div class="log">
          <pre id="log"></pre>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""


def _to_markdown_html(text: str) -> str:
    """Render markdown to HTML if the markdown package is available, else escape."""
    if markdown is not None:
        return markdown.markdown(text, extensions=["fenced_code", "tables"])
    return f"<pre>{html.escape(text)}</pre>"


def _render_page(
    prompt: str = "",
    answer: str | None = None,
    notes: list[str] | None = None,
    debug_log: str | None = None,
) -> HTMLResponse:
    # For the streaming UI, we always render the same base HTML; the JS fills in result areas.
    html_str = HTML_TEMPLATE.replace("__PROMPT__", html.escape(prompt or ""))
    return HTMLResponse(content=html_str)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    print("[ui_app] GET / - serving index page")
    return _render_page()


class StreamingLog(io.StringIO):  # type: ignore[name-defined]
    """
    Log sink that updates the JOBS store as the graph prints output.
    """

    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def write(self, s: str) -> int:  # type: ignore[override]
        n = super().write(s)
        job = JOBS.get(self.job_id)
        if job is not None:
            job["debug_log"] = self.getvalue()
        return n


async def _run_job(job_id: str, prompt: str) -> None:
    JOBS[job_id]["status"] = "running"
    logger = StreamingLog(job_id)
    try:
        full_state = await asyncio.to_thread(run_research_streaming, prompt, logger)
        answer = full_state.get("final_answer") or full_state.get("answer") or ""
        notes = full_state.get("research_notes", [])
        evaluator_score = full_state.get("evaluator_score")
        fix_instructions = full_state.get("fix_instructions", "")

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["answer"] = answer
        JOBS[job_id]["answer_html"] = _to_markdown_html(answer) if answer else ""
        JOBS[job_id]["notes"] = notes
        JOBS[job_id].setdefault("debug_log", logger.getvalue())
        JOBS[job_id]["agent_feedback"] = {
            "evaluator_score": evaluator_score,
            "fix_instructions": fix_instructions,
        }
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        existing_log = JOBS[job_id].get("debug_log", "")
        JOBS[job_id]["debug_log"] = f"{existing_log}\nERROR: {e}".strip()


@app.post("/start")
async def start(request: Request) -> Dict[str, str]:
    data = await request.json()
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "pending",
        "prompt": prompt,
        "answer": "",
        "notes": [],
        "debug_log": "",
    }

    print(f"[ui_app] /start - created job {job_id} for prompt: {prompt!r}")
    asyncio.create_task(_run_job(job_id, prompt))

    return {"job_id": job_id}


@app.get("/status")
async def status(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    # Ensure the result card is shown once any job exists
    job.setdefault("answer", "")
    job.setdefault("notes", [])
    job.setdefault("debug_log", "")
    return job

