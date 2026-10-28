"""Local Streamlit app: paste a news article (text and/or image), get
matching UPSC previous-year questions. Wraps src/agent/pipeline.py.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import html
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.agent.formatting import split_statement_question
from src.agent.pipeline import find_matches, generate_practice_questions

LOG_DIR = Path(__file__).parent / "logs"


@st.cache_resource
def _get_executor() -> ThreadPoolExecutor:
    # One shared pool for the process's lifetime (st.cache_resource, not a
    # plain module-level global) — a Streamlit script reruns top-to-bottom on
    # every interaction, so a bare module-level ThreadPoolExecutor() would
    # spin up a fresh pool (and thread) on every rerun, including every
    # fragment poll tick below.
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="pyq-practice-gen")


def _file_only_log(path: Path):
    """A log callback safe to hand to a background thread — writes to the
    request's log file only, never touches Streamlit (st.status etc. require
    the main script thread's context, which a worker thread doesn't have)."""

    def _log(message: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%H:%M:%S}] {message}\n")

    return _log

st.set_page_config(page_title="PYQ Finder", page_icon="📜", layout="centered")

# ---------------------------------------------------------------------------
# Design system: the source material for this identity is the exam booklets
# themselves — the stamped red exam headers, the boxed booklet-series letter,
# the terse monospace codes stamped on every footer ("XDTG-F-GST/62A"). See
# PLAN.md for the corpus this app draws from; this is that archive's own
# visual language, not a generic dashboard skin.
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --paper: #F1EDE4;
  --paper-raised: #FBF9F4;
  --ink: #1C1B1A;
  --ink-soft: #4A4640;
  --muted: #8A8478;
  --stamp: #A32B1F;
  --stamp-soft: rgba(163, 43, 31, 0.08);
  --seal: #1F3A5F;
  --seal-soft: rgba(31, 58, 95, 0.08);
  --gold: #8C6A1F;
  --gold-soft: rgba(140, 106, 31, 0.08);
  --rule: #D9D2C0;
}

html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--paper) !important;
  color: var(--ink) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMainBlockContainer"] { padding-top: 3rem; max-width: 760px; }

/* -- masthead ------------------------------------------------------------ */
.eyebrow {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--stamp) !important;
  margin: 0 0 0.4rem 0;
}
.wordmark {
  font-family: 'Fraunces', serif !important;
  font-weight: 900 !important;
  font-size: 3rem !important;
  line-height: 1 !important;
  color: var(--ink) !important;
  margin: 0 0 0.6rem 0 !important;
  letter-spacing: -0.01em;
}
.subtitle {
  font-family: 'Source Serif 4', serif !important;
  font-size: 1.05rem !important;
  color: var(--ink-soft) !important;
  max-width: 46ch;
  margin: 0 0 2rem 0;
  line-height: 1.5 !important;
  font-weight: 400 !important;
}

/* -- form ----------------------------------------------------------------- */
[data-testid="stForm"] {
  background: var(--paper-raised);
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 1.75rem 1.75rem 1.5rem;
}
[data-testid="stWidgetLabel"] p {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted) !important;
  font-weight: 500 !important;
}
[data-testid="stTextAreaRootElement"] {
  background: var(--paper) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 4px !important;
}
[data-testid="stTextAreaRootElement"]:focus-within {
  border-color: var(--seal) !important;
  box-shadow: 0 0 0 1px var(--seal) !important;
}
[data-testid="stTextAreaRootElement"] textarea {
  font-family: 'Source Serif 4', serif !important;
  font-size: 0.98rem !important;
  color: var(--ink) !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: var(--paper) !important;
  border: 1px dashed var(--rule) !important;
  border-radius: 4px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
  color: var(--muted) !important;
}
[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--stamp) !important;
  border: none !important;
  border-radius: 4px !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 500 !important;
  padding: 0.55rem 1.4rem !important;
  transition: transform 0.08s ease, background 0.15s ease;
}
[data-testid="stBaseButton-primaryFormSubmit"]:hover { background: #8A241A !important; }
[data-testid="stBaseButton-primaryFormSubmit"]:active { transform: scale(0.97); }
[data-testid="stBaseButton-secondaryFormSubmit"] {
  background: transparent !important;
  border: 1px solid var(--rule) !important;
  border-radius: 4px !important;
  color: var(--ink-soft) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 500 !important;
  padding: 0.55rem 1.1rem !important;
  width: 100%;
  transition: border-color 0.15s ease, color 0.15s ease;
}
[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
  border-color: var(--ink-soft) !important;
  color: var(--ink) !important;
}

/* -- expander -------------------------------------------------------------- */
[data-testid="stExpander"] {
  background: transparent !important;
  border: 1px solid var(--rule) !important;
  border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-soft) !important;
}
[data-testid="stExpanderDetails"] {
  font-family: 'Source Serif 4', serif;
  color: var(--ink-soft);
}
[data-testid="stExpanderDetails"] b, [data-testid="stExpanderDetails"] strong {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}

/* -- tabs ------------------------------------------------------------------ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 1.5rem;
  border-bottom: 1px solid var(--rule);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  padding-bottom: 0.6rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--ink) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  background-color: var(--stamp) !important;
  height: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

/* -- match cards ------------------------------------------------------------ */
.match-card { padding: 1.5rem 0; border-bottom: 1px solid var(--rule); }
.match-card:last-child { border-bottom: none; }
.match-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.7rem;
}
.citation {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.74rem !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted) !important;
}
.stamp-badge {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 500 !important;
  padding: 0.15rem 0.55rem;
  border-radius: 2px;
  transform: rotate(-2deg);
  border: 1.5px solid currentColor;
}
.stamp-badge--direct { color: var(--stamp) !important; background: var(--stamp-soft); }
.stamp-badge--thematic { color: var(--seal) !important; background: var(--seal-soft); }
.stamp-badge--broader { color: var(--gold) !important; background: var(--gold-soft); }
.match-body {
  font-family: 'Source Serif 4', serif !important;
  font-size: 1.05rem !important;
  line-height: 1.6 !important;
  color: var(--ink) !important;
  white-space: pre-line;
  margin-bottom: 0.5rem;
  font-weight: 400 !important;
}
.match-options { margin: 0.4rem 0 0.7rem 0; padding: 0; list-style: none; }
.match-options li {
  font-family: 'Source Serif 4', serif !important;
  font-size: 1rem !important;
  color: var(--ink-soft) !important;
  padding: 0.15rem 0 0.15rem 1.4rem;
  text-indent: -1.4rem;
  font-weight: 400 !important;
}
.match-statements {
  margin: 0.5rem 0 0.6rem 0;
  padding: 0;
  list-style: none;
  counter-reset: statement;
}
.match-statements li {
  font-family: 'Source Serif 4', serif !important;
  font-size: 1.05rem !important;
  line-height: 1.55 !important;
  color: var(--ink) !important;
  font-weight: 400 !important;
  padding: 0.2rem 0 0.2rem 1.9rem;
  position: relative;
}
.match-statements li::before {
  counter-increment: statement;
  content: counter(statement) ".";
  position: absolute;
  left: 0;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.85rem !important;
  color: var(--stamp) !important;
}
.match-outro {
  font-family: 'Source Serif 4', serif !important;
  font-size: 1.05rem !important;
  line-height: 1.6 !important;
  color: var(--ink) !important;
  margin: 0.3rem 0 0.5rem 0;
  font-weight: 400 !important;
}
.match-meta {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.72rem !important;
  color: var(--muted) !important;
  margin-bottom: 0.5rem;
}
.match-reason {
  font-family: 'Source Serif 4', serif !important;
  font-style: italic !important;
  font-size: 0.92rem !important;
  color: var(--ink-soft) !important;
  border-left: 2px solid var(--rule);
  padding-left: 0.7rem;
  font-weight: 400 !important;
}
.empty-state {
  font-family: 'Source Serif 4', serif !important;
  font-style: italic !important;
  color: var(--muted) !important;
  padding: 1rem 0;
}

/* -- practice (AI-generated) questions -------------------------------------
   Deliberately NOT styled like the real-PYQ stamp cards above — dashed
   borders instead of a solid rule, an un-rotated outline badge instead of a
   stamp, so this never reads as an archive record. */
.practice-banner {
  font-family: 'Source Serif 4', serif !important;
  font-style: italic !important;
  font-size: 0.92rem !important;
  color: var(--ink-soft) !important;
  background: var(--paper-raised);
  border: 1px dashed var(--rule);
  border-radius: 4px;
  padding: 0.8rem 1rem;
  margin-bottom: 1.2rem;
  line-height: 1.55;
}
.practice-card { padding: 1.5rem 0; border-bottom: 1px dashed var(--rule); }
.practice-card:last-child { border-bottom: none; }
.generated-badge {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  font-weight: 500 !important;
  padding: 0.15rem 0.55rem;
  border-radius: 10px;
  color: var(--muted) !important;
  border: 1px dashed var(--muted);
  background: transparent;
}
.practice-answer {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.04em;
  color: var(--seal) !important;
  margin: 0.2rem 0 0.5rem 0;
}
.practice-options li.correct { color: var(--seal) !important; font-weight: 600 !important; }
.practice-explanation, .practice-approach {
  font-family: 'Source Serif 4', serif !important;
  font-size: 0.95rem !important;
  color: var(--ink-soft) !important;
  line-height: 1.55 !important;
  margin-top: 0.3rem;
}
.practice-approach b {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
}

@media (prefers-reduced-motion: reduce) {
  [data-testid="stBaseButton-primaryFormSubmit"] { transition: none; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<p class="eyebrow">Previous-Year Question Archive · 2016–2025</p>
<h1 class="wordmark">PYQ Finder</h1>
<p class="subtitle">Paste a news article — text or a photo — and find what the Commission
has already asked on the same trend, not just the same event.</p>
""",
    unsafe_allow_html=True,
)

st.session_state.setdefault("form_run", 0)
run = st.session_state["form_run"]

with st.form(f"article_form_{run}"):
    article_text = st.text_area(
        "Article text",
        height=180,
        placeholder="Paste the news article text here...",
        label_visibility="visible",
        key=f"article_text_{run}",
    )
    uploaded_image = st.file_uploader(
        "...or a photo/screenshot of an article",
        type=["png", "jpg", "jpeg"],
        key=f"uploaded_image_{run}",
    )
    generate_practice = st.checkbox(
        "Also write new practice questions in UPSC's style for this article",
        value=True,
        key=f"generate_practice_{run}",
        help="Original questions authored by the model — not real previous-year questions.",
    )
    col_submit, col_clear = st.columns([3, 1])
    with col_submit:
        submitted = st.form_submit_button("Find similar questions", type="primary")
    with col_clear:
        new_question = st.form_submit_button("New question")

if new_question:
    st.session_state["form_run"] += 1
    for key in (
        "result", "log_file", "practice_future", "practice_requested",
        "generated_prelims", "generated_mains", "practice_error",
    ):
        st.session_state.pop(key, None)
    st.rerun()

if submitted:
    if not article_text.strip() and not uploaded_image:
        st.error("Nothing to search — paste some article text or upload an image first.")
        st.stop()

    image_path = None
    if uploaded_image:
        suffix = Path(uploaded_image.name).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_image.getvalue())
        tmp.close()
        image_path = tmp.name

    # One plain-text log file per request, written to the local drive (not
    # the UI) — see LOG_DIR at the top of this file.
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = log_path.open("a", encoding="utf-8")

    # Streamlit pushes each write() to the browser as the script executes, not
    # only once it finishes — so narrating on_log here turns the ~75s
    # find_matches run from one opaque spinner into a live, step-by-step
    # feed. Practice-question generation is deliberately NOT run in here:
    # find_matches's own output (the real PYQ matches) only depends on
    # itself, so it's shown the moment it's ready rather than making the
    # user wait through generation too — see the background submission
    # right below.
    with st.status("Reading the article and searching previous-year questions...", expanded=True) as status:

        def on_log(message: str) -> None:
            log_file.write(f"[{datetime.now():%H:%M:%S}] {message}\n")
            log_file.flush()
            status.write(message)

        try:
            result = find_matches(
                article_text=article_text.strip() or None,
                image_path=image_path,
                on_log=on_log,
            )
            st.session_state["result"] = result
            st.session_state["log_file"] = str(log_path)
            st.session_state["generated_prelims"] = []
            st.session_state["generated_mains"] = []
            st.session_state.pop("practice_error", None)
            st.session_state["practice_requested"] = generate_practice

            if generate_practice:
                executor = _get_executor()
                st.session_state["practice_future"] = executor.submit(
                    generate_practice_questions,
                    result["context"],
                    result["prelims_matches"],
                    result["mains_matches"],
                    on_log=_file_only_log(log_path),
                )
                status.update(
                    label="Matches ready — writing practice questions in the background",
                    state="complete",
                    expanded=False,
                )
            else:
                st.session_state["practice_future"] = None
                status.update(label="Search complete", state="complete", expanded=False)
        except Exception as e:
            on_log(f"Error — {e}")
            status.update(label="Search failed", state="error", expanded=True)
            st.error(f"The search didn't complete: {e}")
            st.stop()
        finally:
            log_file.close()
            if image_path:
                Path(image_path).unlink(missing_ok=True)

if "log_file" in st.session_state:
    st.caption(f"Full step-by-step log written to `{st.session_state['log_file']}`")

if "result" in st.session_state:
    result = st.session_state["result"]
    context = result["context"]

    with st.expander("What the agent understood from the article"):
        st.markdown(f"**Topic** — {context['topic']}")
        st.markdown(f"**Entities** — {', '.join(context['entities'])}")
        st.markdown(f"**Key facts** — {context['key_facts']}")
        if context.get("augmented_with_search"):
            st.markdown("**Note** — the article was too sparse on its own; augmented with a Google Search lookup.")
        st.markdown("**Search queries used**")
        for q in result["queries"]:
            st.markdown(f"- _{q['angle']}_ — {q['query']}")
        if result["retried"]:
            st.markdown("**Note** — first pass found little, so the search was broadened and retried once.")

    TIER_LABEL = {"direct": "Direct", "thematic": "Thematic", "broader": "Broader"}

    def match_card_html(m: dict) -> str:
        tier = m["tier"] if m["tier"] in TIER_LABEL else "thematic"
        citation = f"UPSC · {m['year']} · {html.escape(m['paper'])} · Q.{m['question_number']}"
        badge = f'<span class="stamp-badge stamp-badge--{tier}">{TIER_LABEL[tier]}</span>'

        # Prefer the LLM-formatted + verified version cached in SQLite (see
        # src/ingest/format_questions.py); fall back to the free local regex
        # splitter, which catches most but not all phrasings.
        formatted = m.get("formatted")
        if formatted:
            intro, statements, outro = formatted["intro"], formatted["statements"], formatted["outro"]
        else:
            split = split_statement_question(m["question_text"])
            intro, statements, outro = split if split else (None, None, None)

        if statements:
            items = "".join(f"<li>{html.escape(s)}</li>" for s in statements)
            body = f"""<div class="match-body">{html.escape(intro)}</div>
<ol class="match-statements">{items}</ol>
{f'<div class="match-outro">{html.escape(outro)}</div>' if outro else ''}"""
        else:
            body = f'<div class="match-body">{html.escape(m["question_text"])}</div>'

        options_html = ""
        if m.get("options"):
            items = "".join(
                f"<li>({html.escape(o['label'])}) {html.escape(o['text'])}</li>" for o in m["options"]
            )
            options_html = f'<ul class="match-options">{items}</ul>'

        meta_html = ""
        if m.get("marks"):
            limit = f" &middot; {m['word_limit']} words" if m.get("word_limit") else ""
            meta_html = f'<div class="match-meta">{m["marks"]} marks{limit}</div>'

        reason = html.escape(m["reason"])
        return f"""
<div class="match-card">
  <div class="match-head">
    <span class="citation">{citation}</span>
    {badge}
  </div>
  {body}
  {options_html}
  {meta_html}
  <div class="match-reason">{reason}</div>
</div>
"""

    def practice_prelims_html(q: dict, index: int) -> str:
        items = "".join(f"<li>{html.escape(s)}</li>" for s in q["statements"])
        body = f"""<div class="match-body">{html.escape(q["intro"])}</div>
<ol class="match-statements">{items}</ol>
<div class="match-outro">{html.escape(q["outro"])}</div>"""

        correct = q["correct_option"]
        opts = "".join(
            f'<li class="{"correct" if o["label"] == correct else ""}">'
            f'({html.escape(o["label"])}) {html.escape(o["text"])}'
            f'{" &check;" if o["label"] == correct else ""}</li>'
            for o in q["options"]
        )
        options_html = f'<ul class="match-options practice-options">{opts}</ul>'
        answer = f"<div class=\"practice-answer\">Answer — ({html.escape(correct)})</div>"
        explanation = f'<div class="practice-explanation">{html.escape(q["explanation"])}</div>'

        return f"""
<div class="match-card practice-card">
  <div class="match-head">
    <span class="citation">Practice · Prelims {index}</span>
    <span class="generated-badge">AI-generated</span>
  </div>
  {body}
  {options_html}
  {answer}
  {explanation}
</div>
"""

    def practice_mains_html(q: dict, index: int) -> str:
        body = f'<div class="match-body">{html.escape(q["question_text"])}</div>'
        meta = f'<div class="match-meta">{html.escape(q["paper"])} &middot; {q["marks"]} marks &middot; {q["word_limit"]} words</div>'
        approach = f'<div class="practice-approach"><b>Approach</b> — {html.escape(q["approach"])}</div>'

        return f"""
<div class="match-card practice-card">
  <div class="match-head">
    <span class="citation">Practice · Mains {index}</span>
    <span class="generated-badge">AI-generated</span>
  </div>
  {body}
  {meta}
  {approach}
</div>
"""

    @st.fragment(run_every=2)
    def _practice_pending() -> None:
        """Polls the background generation future every 2s. Once it
        resolves, stashes the result in session_state and does a full
        st.rerun() — not a fragment-scoped one — so the tab permanently
        switches to the static _practice_ready() view instead of this
        fragment continuing to poll forever after there's nothing left to
        wait for."""
        future = st.session_state.get("practice_future")
        if future is None:
            return  # already resolved by another tick; the graduating rerun below is on its way
        if not future.done():
            st.info("✎ Writing new practice questions in the background — this tab updates automatically.")
            return
        try:
            generated = future.result()
            st.session_state["generated_prelims"] = generated["generated_prelims"]
            st.session_state["generated_mains"] = generated["generated_mains"]
            st.session_state.pop("practice_error", None)
        except Exception as e:
            st.session_state["practice_error"] = str(e)
            st.session_state["generated_prelims"] = []
            st.session_state["generated_mains"] = []
        st.session_state["practice_future"] = None
        st.rerun()

    def _practice_ready() -> None:
        error = st.session_state.get("practice_error")
        if error:
            st.error(f"Practice question generation didn't complete: {error}")
            return
        gen_prelims = st.session_state.get("generated_prelims", [])
        gen_mains = st.session_state.get("generated_mains", [])
        if not gen_prelims and not gen_mains:
            st.markdown('<p class="empty-state">No practice questions generated.</p>', unsafe_allow_html=True)
            return
        st.markdown(
            '<p class="practice-banner">These are original questions written by the model in '
            "UPSC's style and difficulty — not real previous-year questions, and not sourced "
            "from any actual paper. Treat facts in the statements/explanations as unverified "
            "and double-check anything you plan to rely on.</p>",
            unsafe_allow_html=True,
        )
        cards = [practice_prelims_html(q, i + 1) for i, q in enumerate(gen_prelims)]
        cards += [practice_mains_html(q, i + 1) for i, q in enumerate(gen_mains)]
        st.markdown("".join(cards), unsafe_allow_html=True)

    prelims, mains = result["prelims_matches"], result["mains_matches"]
    practice_requested = st.session_state.get("practice_requested", False)

    if not prelims and not mains and not practice_requested:
        st.markdown(
            '<p class="empty-state">No matching previous-year questions found for this article — '
            "try pasting more of it, or a different one.</p>",
            unsafe_allow_html=True,
        )
    else:
        tab_labels = [f"Prelims · {len(prelims)}", f"Mains · {len(mains)}"]
        if practice_requested:
            if st.session_state.get("practice_future") is not None:
                tab_labels.append("Practice · generating…")
            else:
                gen_count = len(st.session_state.get("generated_prelims", [])) + len(
                    st.session_state.get("generated_mains", [])
                )
                tab_labels.append(f"Practice · {gen_count}")
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            if prelims:
                st.markdown("".join(match_card_html(m) for m in prelims), unsafe_allow_html=True)
            else:
                st.markdown('<p class="empty-state">No Prelims matches.</p>', unsafe_allow_html=True)
        with tabs[1]:
            if mains:
                st.markdown("".join(match_card_html(m) for m in mains), unsafe_allow_html=True)
            else:
                st.markdown('<p class="empty-state">No Mains matches.</p>', unsafe_allow_html=True)
        if practice_requested:
            with tabs[2]:
                if st.session_state.get("practice_future") is not None:
                    _practice_pending()
                else:
                    _practice_ready()
