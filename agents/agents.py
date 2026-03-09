"""
agents/agents.py - Multi-Agent System for Math Mentor
Agents: Parser, IntentRouter, Solver, Verifier, Explainer
"""
import json
from typing import Optional
from utils.llm_client import get_llm_response, parse_json_response
from utils.math_tools import solve_equation, differentiate, integrate_expr, compute_limit, safe_eval_math
from rag.rag_pipeline import rag_pipeline
from memory.memory_store import memory_store
from config import config


# ═══════════════════════════════════════════════════════════════
# AGENT 1: PARSER AGENT
# Converts raw input → structured math problem
# ═══════════════════════════════════════════════════════════════

class ParserAgent:
    """
    Cleans OCR/ASR output, identifies topic, variables, constraints.
    Outputs a structured JSON problem spec.
    """
    NAME = "Parser Agent"

    def run(self, raw_input: str, input_type: str = "text") -> dict:
        """
        Args:
            raw_input: raw text (from OCR, ASR, or direct typing)
            input_type: 'text' | 'image_ocr' | 'audio_asr'
        Returns:
            Structured problem dict
        """
        system = """You are a math problem parser. Your job is to:
1. Clean any OCR/ASR artifacts from the input text
2. Identify the mathematical topic
3. Extract variables, constraints, and question type
4. Determine if clarification is needed

You MUST respond with ONLY a valid JSON object, no markdown, no explanation."""

        prompt = f"""Parse this math problem (input_type: {input_type}):

INPUT: {raw_input}

Respond with this exact JSON structure:
{{
  "problem_text": "cleaned, well-formatted version of the problem",
  "topic": "algebra|probability|calculus|linear_algebra|general",
  "sub_topic": "specific sub-topic e.g. quadratic, limits, matrices",
  "question_type": "solve|prove|find|calculate|simplify|explain",
  "variables": ["list", "of", "variables"],
  "given": ["list of given information"],
  "find": "what to find or prove",
  "constraints": ["list of constraints or conditions"],
  "needs_clarification": false,
  "clarification_reason": "",
  "difficulty": "easy|medium|hard",
  "jee_relevant": true
}}"""

        response = get_llm_response(prompt, system=system, json_mode=True)
        parsed = parse_json_response(response)

        # Validate required fields
        parsed.setdefault("problem_text", raw_input)
        parsed.setdefault("topic", "general")
        parsed.setdefault("sub_topic", "")
        parsed.setdefault("variables", [])
        parsed.setdefault("constraints", [])
        parsed.setdefault("needs_clarification", False)
        parsed.setdefault("clarification_reason", "")
        parsed.setdefault("find", "")

        return parsed


# ═══════════════════════════════════════════════════════════════
# AGENT 2: INTENT ROUTER AGENT
# Classifies problem and routes the workflow
# ═══════════════════════════════════════════════════════════════

class IntentRouterAgent:
    """
    Analyzes parsed problem and decides:
    - Which tools to use (sympy, none, etc.)
    - What retrieval strategy to use
    - Whether memory has similar solved problems
    """
    NAME = "Intent Router Agent"

    def run(self, parsed_problem: dict) -> dict:
        """
        Returns routing plan.
        """
        topic = parsed_problem.get("topic", "general")
        sub_topic = parsed_problem.get("sub_topic", "")
        problem_text = parsed_problem.get("problem_text", "")
        question_type = parsed_problem.get("question_type", "solve")

        # Check memory for similar problems
        similar = memory_store.find_similar_problems(problem_text, topic=topic, top_k=2)

        # Determine tools to use based on topic
        tools = []
        if topic == "algebra" or sub_topic in ["quadratic", "equation", "polynomial"]:
            tools.append("sympy_solver")
        if topic == "calculus":
            if "deriv" in problem_text.lower() or "differentiat" in problem_text.lower():
                tools.append("sympy_diff")
            elif "integr" in problem_text.lower():
                tools.append("sympy_integrate")
            elif "limit" in problem_text.lower():
                tools.append("sympy_limit")
            else:
                tools.append("sympy_solver")
        if topic == "linear_algebra":
            tools.append("sympy_matrix")

        # Determine retrieval strategy
        retrieval_filter = topic if topic != "general" else None

        routing_plan = {
            "topic": topic,
            "sub_topic": sub_topic,
            "question_type": question_type,
            "tools_to_use": tools if tools else ["llm_reasoning"],
            "retrieval_topic_filter": retrieval_filter,
            "similar_problems_found": len(similar),
            "similar_problems": similar[:2],
            "estimated_complexity": parsed_problem.get("difficulty", "medium"),
        }

        return routing_plan


# ═══════════════════════════════════════════════════════════════
# AGENT 3: SOLVER AGENT
# Core solving using RAG + tools
# ═══════════════════════════════════════════════════════════════

class SolverAgent:
    """
    Solves the problem using:
    - Retrieved knowledge (RAG)
    - SymPy tools for computation
    - Memory for similar solved problems
    - LLM for reasoning
    """
    NAME = "Solver Agent"

    def run(self, parsed_problem: dict, routing_plan: dict) -> dict:
        """
        Returns solution dict.
        """
        problem_text = parsed_problem.get("problem_text", "")
        topic = routing_plan.get("topic", "general")

        # 1. Retrieve relevant knowledge
        retrieved_chunks = rag_pipeline.retrieve(
            query=problem_text,
            topic=routing_plan.get("retrieval_topic_filter"),
            top_k=4
        )
        context_str = rag_pipeline.build_context_string(retrieved_chunks)

        # 2. Try SymPy tools first
        tool_results = {}
        for tool in routing_plan.get("tools_to_use", []):
            tool_result = self._run_tool(tool, parsed_problem)
            if tool_result.get("success"):
                tool_results[tool] = tool_result

        # 3. Build memory context from similar problems
        memory_context = ""
        similar = routing_plan.get("similar_problems", [])
        if similar:
            memory_lines = ["\n=== SIMILAR SOLVED PROBLEMS FROM MEMORY ==="]
            for s in similar:
                memory_lines.append(f"Problem: {s.get('parsed_problem', {}).get('problem_text', '')}")
                memory_lines.append(f"Solution: {s.get('solution', '')}\n")
            memory_context = "\n".join(memory_lines)

        # 4. Corrections from memory
        corrections = memory_store.get_correction_patterns(topic=topic)
        correction_context = ""
        if corrections:
            corr_lines = ["\n=== PAST CORRECTIONS TO AVOID SAME MISTAKES ==="]
            for c in corrections[-3:]:
                corr_lines.append(f"Problem: {c.get('original_problem', '')}")
                corr_lines.append(f"Wrong approach: {c.get('wrong_answer', '')}")
                corr_lines.append(f"Correct approach: {c.get('correct_answer', '')}\n")
            correction_context = "\n".join(corr_lines)

        # 5. Build tool results string
        tool_str = ""
        if tool_results:
            tool_str = "\n=== COMPUTATIONAL TOOL RESULTS ===\n"
            for tool_name, result in tool_results.items():
                tool_str += f"{tool_name}: {result}\n"

        # 6. Call LLM to solve
        system = """You are an expert JEE math tutor. Solve the problem step-by-step.
Use the retrieved knowledge, tool results, and memory context provided.
Give a clear, structured solution. Show all steps. Give the final answer clearly."""

        prompt = f"""MATH PROBLEM:
{problem_text}

Topic: {topic}
Variables: {parsed_problem.get('variables', [])}
Find: {parsed_problem.get('find', 'the answer')}
Constraints: {parsed_problem.get('constraints', [])}

{context_str}
{tool_str}
{memory_context}
{correction_context}

Solve this step-by-step. End with "FINAL ANSWER: [answer]"."""

        solution_text = get_llm_response(prompt, system=system, max_tokens=2000)

        # Extract final answer
        final_answer = ""
        if "FINAL ANSWER:" in solution_text:
            final_answer = solution_text.split("FINAL ANSWER:")[-1].strip()

        return {
            "solution_text": solution_text,
            "final_answer": final_answer,
            "retrieved_chunks": retrieved_chunks,
            "tool_results": tool_results,
            "context_used": context_str,
        }

    def _run_tool(self, tool_name: str, parsed_problem: dict) -> dict:
        """Run a specific computational tool."""
        problem_text = parsed_problem.get("problem_text", "")
        variables = parsed_problem.get("variables", ["x"])
        var = variables[0] if variables else "x"

        try:
            if tool_name == "sympy_solver":
                return solve_equation(problem_text, var_str=var)
            elif tool_name == "sympy_diff":
                return differentiate(problem_text, var_str=var)
            elif tool_name == "sympy_integrate":
                return integrate_expr(problem_text, var_str=var)
            elif tool_name == "sympy_limit":
                return compute_limit(problem_text, var_str=var)
            elif tool_name == "sympy_matrix":
                return {"success": False, "note": "Matrix operations need explicit matrix data"}
            else:
                return {"success": False}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# AGENT 4: VERIFIER / CRITIC AGENT
# Checks correctness, units, edge cases
# ═══════════════════════════════════════════════════════════════

class VerifierAgent:
    """
    Critically reviews the solution for:
    - Mathematical correctness
    - Domain/unit validity
    - Edge cases
    - Completeness
    Returns confidence score and issues found.
    """
    NAME = "Verifier Agent"

    def run(self, parsed_problem: dict, solution: dict) -> dict:
        """
        Returns verification result.
        """
        system = """You are a strict math exam checker. Review the solution for correctness.
You MUST respond with ONLY valid JSON, no markdown."""

        prompt = f"""Review this math solution for correctness.

PROBLEM: {parsed_problem.get('problem_text', '')}
Topic: {parsed_problem.get('topic', '')}
Constraints: {parsed_problem.get('constraints', [])}

SOLUTION:
{solution.get('solution_text', '')}

FINAL ANSWER CLAIMED: {solution.get('final_answer', '')}

Check for:
1. Mathematical errors in steps
2. Correct application of formulas
3. Domain violations (e.g., sqrt of negative, log of zero)
4. Missing cases or edge cases
5. Arithmetic errors
6. Unit/dimension consistency

Respond with this JSON:
{{
  "is_correct": true,
  "confidence": 0.9,
  "issues_found": [],
  "missing_cases": [],
  "domain_violations": [],
  "arithmetic_errors": [],
  "formula_errors": [],
  "suggestion": "overall assessment",
  "needs_human_review": false,
  "corrected_answer": ""
}}

Set needs_human_review=true if confidence < {config.VERIFIER_CONFIDENCE_THRESHOLD}."""

        response = get_llm_response(prompt, system=system, json_mode=True)
        result = parse_json_response(response)

        # Set defaults
        result.setdefault("is_correct", True)
        result.setdefault("confidence", 0.8)
        result.setdefault("issues_found", [])
        result.setdefault("needs_human_review", False)
        result.setdefault("suggestion", "Solution looks correct.")
        result.setdefault("corrected_answer", "")

        # Force human review if confidence too low
        if result.get("confidence", 1.0) < config.VERIFIER_CONFIDENCE_THRESHOLD:
            result["needs_human_review"] = True

        return result


# ═══════════════════════════════════════════════════════════════
# AGENT 5: EXPLAINER / TUTOR AGENT
# Produces student-friendly step-by-step explanation
# ═══════════════════════════════════════════════════════════════

class ExplainerAgent:
    """
    Transforms the solution into a student-friendly, educational explanation.
    Includes: key concepts used, step rationale, tips, common mistakes.
    """
    NAME = "Explainer Agent"

    def run(self, parsed_problem: dict, solution: dict, verifier_output: dict) -> dict:
        """
        Returns structured explanation.
        """
        system = """You are a friendly, encouraging JEE math tutor.
Explain solutions clearly for a student. Make it educational and engaging.
You MUST respond with ONLY valid JSON."""

        prompt = f"""Create a student-friendly explanation for this math problem and solution.

PROBLEM: {parsed_problem.get('problem_text', '')}
Topic: {parsed_problem.get('topic', '')}
Sub-topic: {parsed_problem.get('sub_topic', '')}

SOLUTION (internal):
{solution.get('solution_text', '')}

VERIFIER NOTES:
- Correct: {verifier_output.get('is_correct', True)}
- Issues: {verifier_output.get('issues_found', [])}
- Suggestion: {verifier_output.get('suggestion', '')}

Create an explanation with:
{{
  "concept_overview": "2-3 sentences explaining the key concept",
  "steps": [
    {{
      "step_number": 1,
      "title": "Step title",
      "explanation": "What we do and WHY",
      "math": "The actual mathematical work",
      "tip": "optional tip or trick"
    }}
  ],
  "key_formulas_used": ["formula1", "formula2"],
  "common_mistakes": ["mistake to avoid 1", "mistake 2"],
  "final_answer": "clean final answer",
  "memory_tip": "mnemonic or tip to remember this type of problem"
}}"""

        response = get_llm_response(prompt, system=system, max_tokens=2500, json_mode=True)
        result = parse_json_response(response)

        result.setdefault("concept_overview", "")
        result.setdefault("steps", [])
        result.setdefault("key_formulas_used", [])
        result.setdefault("common_mistakes", [])
        result.setdefault("final_answer", solution.get("final_answer", ""))
        result.setdefault("memory_tip", "")

        return result


# ═══════════════════════════════════════════════════════════════
# AGENT 6 (BONUS): GUARDRAIL AGENT
# ═══════════════════════════════════════════════════════════════

class GuardrailAgent:
    """
    Ensures input is a valid math problem (not harmful, not off-topic).
    """
    NAME = "Guardrail Agent"

    def check(self, raw_input: str) -> dict:
        system = """You are a content filter for a JEE math tutoring app.
Check if the input is a valid math problem. Respond with ONLY JSON."""

        prompt = f"""Is this a valid math problem appropriate for a JEE math tutor?

INPUT: {raw_input}

Respond:
{{
  "is_valid": true,
  "is_math": true,
  "is_appropriate": true,
  "reason": "explanation if invalid"
}}"""

        response = get_llm_response(prompt, system=system, json_mode=True)
        result = parse_json_response(response)
        result.setdefault("is_valid", True)
        result.setdefault("is_math", True)
        return result


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# Runs all agents in sequence
# ═══════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """
    Orchestrates the full multi-agent pipeline.
    """

    def __init__(self):
        self.guardrail = GuardrailAgent()
        self.parser = ParserAgent()
        self.router = IntentRouterAgent()
        self.solver = SolverAgent()
        self.verifier = VerifierAgent()
        self.explainer = ExplainerAgent()

    def run_pipeline(self, raw_input: str, input_type: str = "text") -> dict:
        """
        Full pipeline run. Returns complete result dict.
        """
        trace = []  # Agent execution trace

        # ── Step 0: Guardrail check
        trace.append({"agent": GuardrailAgent.NAME, "status": "running"})
        guard = self.guardrail.check(raw_input)
        trace[-1]["status"] = "done"
        trace[-1]["output"] = guard
        if not guard.get("is_valid", True) or not guard.get("is_math", True):
            return {
                "success": False,
                "error": f"Invalid input: {guard.get('reason', 'Not a math problem')}",
                "trace": trace,
            }

        # ── Step 1: Parse
        trace.append({"agent": ParserAgent.NAME, "status": "running"})
        parsed_problem = self.parser.run(raw_input, input_type=input_type)
        trace[-1]["status"] = "done"
        trace[-1]["output"] = parsed_problem

        # ── Check if needs clarification
        if parsed_problem.get("needs_clarification"):
            return {
                "success": False,
                "needs_clarification": True,
                "clarification_reason": parsed_problem.get("clarification_reason", "Problem is ambiguous"),
                "parsed_problem": parsed_problem,
                "trace": trace,
            }

        # ── Step 2: Route
        trace.append({"agent": IntentRouterAgent.NAME, "status": "running"})
        routing_plan = self.router.run(parsed_problem)
        trace[-1]["status"] = "done"
        trace[-1]["output"] = routing_plan

        # ── Step 3: Solve
        trace.append({"agent": SolverAgent.NAME, "status": "running"})
        solution = self.solver.run(parsed_problem, routing_plan)
        trace[-1]["status"] = "done"
        trace[-1]["output"] = {
            "final_answer": solution.get("final_answer"),
            "num_chunks_retrieved": len(solution.get("retrieved_chunks", [])),
            "tools_used": list(solution.get("tool_results", {}).keys()),
        }

        # ── Step 4: Verify
        trace.append({"agent": VerifierAgent.NAME, "status": "running"})
        verifier_output = self.verifier.run(parsed_problem, solution)
        trace[-1]["status"] = "done"
        trace[-1]["output"] = {
            "is_correct": verifier_output.get("is_correct"),
            "confidence": verifier_output.get("confidence"),
            "needs_human_review": verifier_output.get("needs_human_review"),
        }

        # ── Step 5: Explain
        trace.append({"agent": ExplainerAgent.NAME, "status": "running"})
        explanation = self.explainer.run(parsed_problem, solution, verifier_output)
        trace[-1]["status"] = "done"
        trace[-1]["output"] = {"steps_count": len(explanation.get("steps", []))}

        return {
            "success": True,
            "parsed_problem": parsed_problem,
            "routing_plan": routing_plan,
            "solution": solution,
            "verifier_output": verifier_output,
            "explanation": explanation,
            "trace": trace,
            "needs_human_review": verifier_output.get("needs_human_review", False),
        }


# Singleton
orchestrator = AgentOrchestrator()
