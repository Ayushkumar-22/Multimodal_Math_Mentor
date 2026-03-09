"""
app.py - Math Mentor AI - Main Streamlit Application
"""
import streamlit as st
import json
import os
import sys
import io
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import config

# ─── Page Config (MUST be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="Math Mentor AI",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .agent-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .agent-running { border-left-color: #ffc107; }
    .agent-done { border-left-color: #28a745; }
    .agent-error { border-left-color: #dc3545; }
    .chunk-card {
        background: #e8f4f8;
        border: 1px solid #bee5eb;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .step-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .final-answer {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
    .hitl-warning {
        background: #fff3cd;
        border: 2px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        margin-top: 4px;
    }
    .memory-badge {
        background: #e3f2fd;
        border: 1px solid #90caf9;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .tag-algebra { background: #d4edda; color: #155724; }
    .tag-calculus { background: #cce5ff; color: #004085; }
    .tag-probability { background: #fff3cd; color: #856404; }
    .tag-linear_algebra { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ──────────────────────────────
def init_session_state():
    defaults = {
        "result": None,
        "session_id": None,
        "ocr_text": "",
        "ocr_confidence": 0.0,
        "asr_text": "",
        "asr_confidence": 0.0,
        "input_confirmed": False,
        "feedback_given": False,
        "hitl_approved": False,
        "processing": False,
        "rag_initialized": False,
        "history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ─── Lazy imports (avoid loading heavy models at startup) ──────
@st.cache_resource(show_spinner=False)
def get_orchestrator():
    from agents.agents import orchestrator
    return orchestrator

@st.cache_resource(show_spinner=False)
def get_memory():
    from memory.memory_store import memory_store
    return memory_store

@st.cache_resource(show_spinner=False)
def init_rag():
    from rag.rag_pipeline import rag_pipeline
    rag_pipeline.initialize()
    return rag_pipeline


# ─── Helper Functions ──────────────────────────────────────────

def process_image_input(image_bytes: bytes):
    """Handle image upload and OCR."""
    from utils.ocr import extract_text_from_image, preprocess_image
    preprocessed = preprocess_image(image_bytes)
    text, confidence = extract_text_from_image(preprocessed)
    st.session_state.ocr_text = text
    st.session_state.ocr_confidence = confidence
    return text, confidence


def process_audio_input(audio_bytes: bytes, file_format: str = "wav"):
    """Handle audio upload and ASR."""
    from utils.asr import transcribe_audio, normalize_math_transcript
    text, confidence = transcribe_audio(audio_bytes, file_format)
    text = normalize_math_transcript(text)
    st.session_state.asr_text = text
    st.session_state.asr_confidence = confidence
    return text, confidence


def run_pipeline(problem_text: str, input_type: str):
    """Run the full agent pipeline."""
    orchestrator = get_orchestrator()
    memory = get_memory()
    init_rag()

    with st.spinner("🤖 Agents working..."):
        result = orchestrator.run_pipeline(problem_text, input_type=input_type)

    if result.get("success"):
        # Save to memory
        session_data = {
            "input_type": input_type,
            "raw_input": problem_text,
            "parsed_problem": result.get("parsed_problem", {}),
            "retrieved_chunks": [c.get("title") for c in result.get("solution", {}).get("retrieved_chunks", [])],
            "solution": result.get("solution", {}).get("solution_text", ""),
            "explanation": str(result.get("explanation", {})),
            "verifier_output": result.get("verifier_output", {}),
            "topic": result.get("parsed_problem", {}).get("topic", "unknown"),
        }
        session_id = memory.save_session(session_data)
        st.session_state.session_id = session_id
        st.session_state.result = result
        st.session_state.history.append({
            "id": session_id,
            "problem": problem_text[:80] + "..." if len(problem_text) > 80 else problem_text,
            "topic": result.get("parsed_problem", {}).get("topic", "unknown"),
        })
    else:
        st.session_state.result = result

    return result


def render_confidence_bar(confidence: float, label: str):
    """Render a colored confidence bar."""
    color = "#28a745" if confidence >= 0.75 else "#ffc107" if confidence >= 0.5 else "#dc3545"
    pct = int(confidence * 100)
    st.markdown(f"""
    <div style="margin: 0.3rem 0;">
        <span style="font-size:0.85rem; color:#666;">{label}: <strong>{pct}%</strong></span>
        <div style="background:#e9ecef; border-radius:4px; height:8px; margin-top:4px;">
            <div style="width:{pct}%; background:{color}; height:8px; border-radius:4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_topic_tag(topic: str):
    css_class = f"tag-{topic}" if topic in ["algebra", "calculus", "probability", "linear_algebra"] else "tag-algebra"
    st.markdown(f'<span class="tag {css_class}">{topic.replace("_", " ").title()}</span>', unsafe_allow_html=True)


# ─── MAIN LAYOUT ───────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🧮 Math Mentor AI</h1>
    <p>JEE-Level Math Problem Solver · RAG + Multi-Agent · Self-Learning</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # API Key input if not in env
    # if not config.ANTHROPIC_API_KEY and not config.OPENAI_API_KEY:
    #     st.markdown("### 🔑 API Key Required")
    #     api_key = st.text_input("Enter Anthropic API Key", type="password", key="api_key_input")
    #     if api_key:
    #         os.environ["ANTHROPIC_API_KEY"] = api_key
    #         config.ANTHROPIC_API_KEY = api_key
    #         st.success("✅ API Key set!")

    # provider_display = f"{'Claude (Anthropic)' if config.LLM_PROVIDER == 'anthropic' else 'GPT-4 (OpenAI)'}"
    # st.info(f"🤖 Using: **{provider_display}**\nModel: `{config.LLM_MODEL}`")

    st.markdown("---")
    st.markdown("## 📊 Session Stats")

    memory = get_memory()
    stats = memory.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Solved", stats["total_sessions"])
        st.metric("Correct", stats["correct_feedback"])
    with col2:
        st.metric("Incorrect", stats["incorrect_feedback"])
        st.metric("Corrections", stats["total_corrections"])

    if stats["topics_distribution"]:
        st.markdown("**Topics Breakdown:**")
        for topic, count in stats["topics_distribution"].items():
            st.markdown(f"- {topic}: **{count}**")

    st.markdown("---")
    st.markdown("## Recent History")
    recent = memory.get_recent_sessions(5)
    if recent:
        for s in reversed(recent):
            st.markdown(f"**#{s['id']}** - {s.get('topic', '?')}")
            p_text = s.get('parsed_problem', {}).get('problem_text', '')
            if p_text:
                st.caption(p_text[:60] + "...")
    else:
        st.caption("No solved problems yet.")

    st.markdown("---")
    if st.button("🗑️ Clear Memory", use_container_width=True):
        import shutil
        if os.path.exists(config.MEMORY_DB_PATH):
            os.remove(config.MEMORY_DB_PATH)
        st.success("Memory cleared!")
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ═══════════════════════════════════════════════════════════════

# Input Mode Selector
st.markdown("### Input Mode")
input_mode = st.radio(
    "Choose how to input your problem:",
    ["✏️ Text", "🖼️ Image (OCR)", "🎤 Audio (ASR)"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ─── TABS for input ────────────────────────────────────────────
problem_text = ""
input_type = "text"
ready_to_solve = False

# ────────────────────────────────────────────────────────────────
# INPUT MODE: TEXT
# ────────────────────────────────────────────────────────────────
if input_mode == "✏️ Text":
    st.markdown("### Type Your Math Problem")
    st.caption("Supports JEE-style algebra, probability, calculus, and linear algebra.")

    sample_problems = {
        "Select a sample...": "",
        "Quadratic Equation": "Solve: 2x² - 5x + 3 = 0. Find all values of x.",
        "Probability": "Two cards are drawn from a deck of 52. What is the probability that both are aces?",
        "Derivative": "Find the derivative of f(x) = x³ · sin(x) using the product rule.",
        "Integration": "Evaluate ∫(x² + 3x + 2)dx from 0 to 2.",
        "Limit": "Find: lim(x→0) [sin(3x) / (2x)]",
        "Matrix": "Find the determinant of the matrix A = [[1,2,3],[4,5,6],[7,8,9]]",
        "Sequence": "Find the sum of the first 20 terms of the AP: 3, 7, 11, 15, ...",
        "Probability Bayes": "A bag has 3 red and 5 blue balls. Two balls are drawn without replacement. Find P(both red).",
        "Optimization": "Find the maximum value of f(x) = -x² + 6x - 5. What is x at maximum?",
    }

    selected_sample = st.selectbox("Try a sample problem:", list(sample_problems.keys()))
    sample_text = sample_problems[selected_sample]

    user_text = st.text_area(
        "Enter your math problem:",
        value=sample_text,
        height=120,
        placeholder="e.g., Solve x² - 5x + 6 = 0, or find d/dx of sin(x)·cos(x)",
    )

    if user_text.strip():
        problem_text = user_text.strip()
        input_type = "text"
        ready_to_solve = True


# ────────────────────────────────────────────────────────────────
# INPUT MODE: IMAGE
# ────────────────────────────────────────────────────────────────
elif input_mode == "🖼️ Image (OCR)":
    st.markdown("### Upload Image of Math Problem")
    st.caption("Upload a JPG/PNG photo or screenshot of your math problem.")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image_bytes = uploaded_file.read()

        col_img, col_ocr = st.columns([1, 1])

        with col_img:
            st.markdown("**📷 Uploaded Image:**")
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, use_column_width=True)

        with col_ocr:
            st.markdown("**OCR Extraction:**")

            if st.button("Extract Text from Image", use_container_width=True):
                with st.spinner("Running OCR..."):
                    ocr_text, ocr_conf = process_image_input(image_bytes)

            if st.session_state.ocr_text:
                render_confidence_bar(st.session_state.ocr_confidence, "OCR Confidence")

                # HITL trigger for low confidence
                if st.session_state.ocr_confidence < config.OCR_CONFIDENCE_THRESHOLD:
                    st.markdown("""
                    <div class="hitl-warning">
                        ⚠️ <strong>HITL Triggered:</strong> Low OCR confidence detected.
                        Please review and correct the extracted text below.
                    </div>
                    """, unsafe_allow_html=True)

                edited_text = st.text_area(
                    "Review & correct extracted text:",
                    value=st.session_state.ocr_text,
                    height=150,
                    key="ocr_edit"
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Confirm Text", use_container_width=True, type="primary"):
                        problem_text = edited_text
                        input_type = "image_ocr"
                        ready_to_solve = True
                        st.session_state.input_confirmed = True
                with col_b:
                    if st.button("🔄 Re-extract", use_container_width=True):
                        st.session_state.ocr_text = ""
                        st.rerun()

                if st.session_state.input_confirmed:
                    problem_text = st.session_state.get("ocr_edit", st.session_state.ocr_text)
                    input_type = "image_ocr"
                    ready_to_solve = True

    if not uploaded_file:
        st.info("**Tip:** Take a clear photo of your textbook problem or handwritten notes.")


# ────────────────────────────────────────────────────────────────
# INPUT MODE: AUDIO
# ────────────────────────────────────────────────────────────────
elif input_mode == "🎤 Audio (ASR)":
    st.markdown("### Upload Audio of Your Math Question")
    st.caption("Upload a WAV/MP3 file, or record audio. Speak clearly with math terms.")

    uploaded_audio = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "m4a", "ogg"],
        label_visibility="collapsed"
    )

    st.markdown("**Or type math phrases like:**")
    st.code("'square root of 16 plus x raised to the power 2'")

    if uploaded_audio:
        audio_bytes = uploaded_audio.read()
        file_ext = uploaded_audio.name.split(".")[-1].lower()

        st.audio(audio_bytes, format=f"audio/{file_ext}")

        if st.button("Transcribe Audio", use_container_width=True):
            with st.spinner(f"Transcribing with Whisper ({config.WHISPER_MODEL})..."):
                asr_text, asr_conf = process_audio_input(audio_bytes, file_format=file_ext)

        if st.session_state.asr_text:
            render_confidence_bar(st.session_state.asr_confidence, "Transcription Confidence")

            # HITL trigger
            if st.session_state.asr_confidence < config.ASR_CONFIDENCE_THRESHOLD:
                st.markdown("""
                <div class="hitl-warning">
                    ⚠️ <strong>HITL Triggered:</strong> Low transcription confidence.
                    Please review the transcript below.
                </div>
                """, unsafe_allow_html=True)

            edited_asr = st.text_area(
                "Review & correct transcript:",
                value=st.session_state.asr_text,
                height=120,
                key="asr_edit"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Confirm Transcript", use_container_width=True, type="primary"):
                    problem_text = edited_asr
                    input_type = "audio_asr"
                    ready_to_solve = True
                    st.session_state.input_confirmed = True
            with col_b:
                if st.button("🔄 Re-transcribe", use_container_width=True):
                    st.session_state.asr_text = ""
                    st.rerun()

            if st.session_state.input_confirmed:
                problem_text = st.session_state.get("asr_edit", st.session_state.asr_text)
                input_type = "audio_asr"
                ready_to_solve = True


# ═══════════════════════════════════════════════════════════════
# SOLVE BUTTON
# ═══════════════════════════════════════════════════════════════
st.markdown("---")

col_solve, col_clear = st.columns([3, 1])
with col_solve:
    solve_clicked = st.button(
        "Solve Problem",
        disabled=not ready_to_solve,
        use_container_width=True,
        type="primary",
    )
with col_clear:
    if st.button("🔄 Reset", use_container_width=True):
        for key in ["result", "session_id", "ocr_text", "asr_text",
                    "input_confirmed", "feedback_given", "hitl_approved"]:
            st.session_state[key] = None if "result" in key or "id" in key else \
                                    "" if "text" in key else \
                                    0.0 if "confidence" in key else False
        st.rerun()

if solve_clicked and problem_text:
    result = run_pipeline(problem_text, input_type)


# ═══════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════
result = st.session_state.get("result")

if result:

    # ── Error / Clarification needed ──
    if not result.get("success"):
        if result.get("needs_clarification"):
            st.warning(f"❓ **Clarification Needed:** {result.get('clarification_reason', '')}")
            parsed = result.get("parsed_problem", {})
            if parsed.get("problem_text"):
                st.info(f"Interpreted as: {parsed['problem_text']}")
        else:
            st.error(f"{result.get('error', 'Something went wrong')}")
    else:
        # ── HITL Warning ──
        if result.get("needs_human_review"):
            st.markdown("""
            <div class="hitl-warning">
                ⚠️ <strong>Human Review Requested:</strong> The verifier is not fully confident.
                Please review the solution carefully before accepting it.
            </div>
            """, unsafe_allow_html=True)

        # ═══ TABS for results ════════════════════════════════════
        tab_explain, tab_solution, tab_agents, tab_rag, tab_verify = st.tabs([
            "Explanation",
            "Full Solution",
            "Agent Trace",
            "Retrieved Context",
            "Verification",
        ])

        parsed = result.get("parsed_problem", {})
        solution = result.get("solution", {})
        explanation = result.get("explanation", {})
        verifier = result.get("verifier_output", {})
        routing = result.get("routing_plan", {})


        # ── TAB 1: EXPLANATION ────────────────────────────────
        with tab_explain:
            # Problem summary row
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**Problem:** {parsed.get('problem_text', '')}")
            with col2:
                render_topic_tag(parsed.get("topic", "general"))
            with col3:
                diff = parsed.get("difficulty", "medium")
                diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(diff, "⚪")
                st.markdown(f"{diff_color} **{diff.title()}**")

            # Concept overview
            if explanation.get("concept_overview"):
                st.info(f"**Concept:** {explanation['concept_overview']}")

            # Memory reuse badge
            if routing.get("similar_problems_found", 0) > 0:
                st.markdown(f'<span class="memory-badge">{routing["similar_problems_found"]} similar problem(s) found in memory</span>', unsafe_allow_html=True)

            # Step-by-step
            if explanation.get("steps"):
                st.markdown("### 📝 Step-by-Step Solution")
                for step in explanation["steps"]:
                    with st.expander(f"Step {step.get('step_number', '?')}: {step.get('title', '')}", expanded=True):
                        st.markdown(f"**What we do:** {step.get('explanation', '')}")
                        if step.get("math"):
                            st.code(step["math"], language="")
                        if step.get("tip"):
                            st.success(f"💡 Tip: {step['tip']}")

            # Key formulas
            if explanation.get("key_formulas_used"):
                st.markdown("### Key Formulas Used")
                for f in explanation["key_formulas_used"]:
                    st.markdown(f"- `{f}`")

            # Final answer
            final_ans = explanation.get("final_answer") or solution.get("final_answer", "")
            if final_ans:
                st.markdown(f"""<div class="final-answer">🎯 Final Answer: {final_ans}</div>""", unsafe_allow_html=True)

            # Common mistakes
            if explanation.get("common_mistakes"):
                st.markdown("### ⚠️ Common Mistakes to Avoid")
                for m in explanation["common_mistakes"]:
                    st.markdown(f"- {m}")

            # Memory tip
            if explanation.get("memory_tip"):
                st.success(f"**Remember:** {explanation['memory_tip']}")


        # ── TAB 2: FULL SOLUTION ──────────────────────────────
        with tab_solution:
            st.markdown("### Complete Solution")
            solution_text = solution.get("solution_text", "No solution generated.")
            st.markdown(solution_text)

            if solution.get("tool_results"):
                st.markdown("### Computational Tool Results")
                for tool_name, res in solution["tool_results"].items():
                    st.markdown(f"**{tool_name}:**")
                    st.json(res)


        # ── TAB 3: AGENT TRACE ────────────────────────────────
        with tab_agents:
            st.markdown("### Agent Execution Trace")
            st.caption("Shows which agents ran, in what order, and what they produced.")

            trace = result.get("trace", [])
            for i, step in enumerate(trace):
                agent_name = step.get("agent", f"Agent {i+1}")
                status = step.get("status", "done")
                output = step.get("output", {})

                css_class = f"agent-{status}"
                icon = "✅" if status == "done" else "⚡" if status == "running" else "❌"

                with st.expander(f"{icon} {agent_name}", expanded=(i < 3)):
                    st.markdown(f"**Status:** {status.upper()}")
                    if output and isinstance(output, dict):
                        # Show key outputs
                        for k, v in output.items():
                            if v and v != [] and v != {}:
                                if isinstance(v, (str, int, float, bool)):
                                    st.markdown(f"- **{k}:** {v}")
                                elif isinstance(v, list) and len(v) < 6:
                                    st.markdown(f"- **{k}:** {', '.join(str(x) for x in v)}")
                                elif isinstance(v, dict):
                                    with st.container():
                                        st.json(v)

            # Routing summary
            st.markdown("### Routing Plan")
            routing_display = {k: v for k, v in routing.items() if k != "similar_problems"}
            st.json(routing_display)


        # ── TAB 4: RAG CONTEXT ────────────────────────────────
        with tab_rag:
            st.markdown("### Retrieved Knowledge Chunks")
            st.caption("These are the relevant formulas and theory retrieved from the knowledge base.")

            retrieved = solution.get("retrieved_chunks", [])
            if retrieved:
                for chunk in retrieved:
                    score = chunk.get("score", 0)
                    score_color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                    with st.expander(f"{score_color} [{chunk['topic'].upper()}] {chunk['title']} — Relevance: {score:.1%}"):
                        st.markdown(chunk["content"])
            else:
                st.info("No chunks retrieved (or RAG not initialized). Install dependencies for full RAG support.")

            # Memory context
            similar = routing.get("similar_problems", [])
            if similar:
                st.markdown("### Memory: Similar Past Problems")
                for s in similar:
                    with st.expander(f"#{s.get('id', '?')} - {s.get('topic', '?')}"):
                        p = s.get('parsed_problem', {})
                        st.markdown(f"**Problem:** {p.get('problem_text', '')}")
                        st.markdown(f"**Solution:** {s.get('solution', '')[:300]}...")
                        st.markdown(f"**Feedback:** {s.get('feedback', 'none')}")


        # ── TAB 5: VERIFICATION ───────────────────────────────
        with tab_verify:
            st.markdown("### Solution Verification")

            conf = verifier.get("confidence", 0.8)
            is_correct = verifier.get("is_correct", True)
            render_confidence_bar(conf, "Verifier Confidence")

            status_icon = "✅" if is_correct else "⚠️"
            status_text = "Likely Correct" if is_correct else "Potential Issues Found"
            st.markdown(f"**{status_icon} Status:** {status_text}")

            if verifier.get("issues_found"):
                st.markdown("**Issues Found:**")
                for issue in verifier["issues_found"]:
                    st.warning(f"- {issue}")

            if verifier.get("domain_violations"):
                st.error(f"**Domain violations:** {', '.join(verifier['domain_violations'])}")

            if verifier.get("arithmetic_errors"):
                st.error(f"**Arithmetic errors:** {', '.join(verifier['arithmetic_errors'])}")

            if verifier.get("suggestion"):
                st.info(f"**Verifier notes:** {verifier['suggestion']}")

            if verifier.get("corrected_answer"):
                st.markdown(f"**Suggested correction:** {verifier['corrected_answer']}")

            if verifier.get("needs_human_review"):
                st.markdown("""
                <div class="hitl-warning">
                    👤 <strong>HITL:</strong> Human review recommended for this solution.
                </div>
                """, unsafe_allow_html=True)


        # ═══════════════════════════════════════════════════════
        # FEEDBACK SECTION
        # ═══════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### Feedback")

        if not st.session_state.feedback_given:
            st.caption("Was this solution correct? Your feedback helps the system learn!")

            col_fb1, col_fb2 = st.columns(2)

            with col_fb1:
                if st.button("Correct!", use_container_width=True, type="primary"):
                    memory = get_memory()
                    if st.session_state.session_id:
                        memory.update_feedback(st.session_state.session_id, "correct")
                    st.session_state.feedback_given = True
                    st.success("Thank you! Stored as correct solution.")
                    st.rerun()

            with col_fb2:
                if st.button("Incorrect", use_container_width=True):
                    st.session_state.show_correction = True

            if st.session_state.get("show_correction"):
                correction = st.text_area(
                    "Please provide the correct answer / correction:",
                    placeholder="e.g., The correct answer is x = 3, not x = 2 because...",
                    key="correction_input"
                )
                if st.button("Submit Correction", type="primary"):
                    memory = get_memory()
                    if st.session_state.session_id:
                        memory.update_feedback(
                            st.session_state.session_id,
                            "incorrect",
                            correction=correction
                        )
                    st.session_state.feedback_given = True
                    st.session_state.show_correction = False
                    st.success("Correction saved! The system will learn from this.")
                    st.rerun()
        else:
            st.success("Feedback recorded. The system will use this to improve.")


# ─── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#aaa; font-size:0.8rem; padding: 1rem;">
    🧮 Math Mentor AI · Built with Streamlit + ChromaDB · JEE Math Solver
</div>
""", unsafe_allow_html=True)
