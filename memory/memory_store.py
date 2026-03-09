"""
memory/memory_store.py - Persistent memory layer for Math Mentor
Stores solved problems, corrections, and patterns for self-learning.
"""
import json
import os
import uuid
from datetime import datetime
from typing import Optional
from config import config


class MemoryStore:
    """
    JSON-based persistent memory store.
    Stores: input, parsed problem, retrieved context, answer, feedback.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.MEMORY_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"sessions": [], "corrections": [], "patterns": {}}

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(self._data, f, indent=2, default=str)

    def save_session(self, session: dict) -> str:
        """Save a complete problem-solving session."""
        session_id = str(uuid.uuid4())[:8]
        record = {
            "id": session_id,
            "timestamp": datetime.now().isoformat(),
            "input_type": session.get("input_type", "text"),
            "raw_input": session.get("raw_input", ""),
            "parsed_problem": session.get("parsed_problem", {}),
            "retrieved_chunks": session.get("retrieved_chunks", []),
            "solution": session.get("solution", ""),
            "explanation": session.get("explanation", ""),
            "verifier_output": session.get("verifier_output", {}),
            "feedback": session.get("feedback", None),  # "correct" / "incorrect"
            "correction": session.get("correction", None),
            "topic": session.get("topic", "unknown"),
        }
        self._data["sessions"].append(record)
        self._save()
        return session_id

    def update_feedback(self, session_id: str, feedback: str, correction: str = None):
        """Update user feedback for a session."""
        for s in self._data["sessions"]:
            if s["id"] == session_id:
                s["feedback"] = feedback
                s["correction"] = correction
                s["feedback_timestamp"] = datetime.now().isoformat()
                break

        # Store correction as a learning pattern
        if feedback == "incorrect" and correction:
            self._store_correction_pattern(session_id, correction)

        self._save()

    def _store_correction_pattern(self, session_id: str, correction: str):
        """Store human correction as a reusable pattern."""
        session = self.get_session(session_id)
        if session:
            pattern = {
                "id": str(uuid.uuid4())[:8],
                "problem_topic": session.get("topic", "unknown"),
                "original_problem": session.get("parsed_problem", {}).get("problem_text", ""),
                "wrong_answer": session.get("solution", ""),
                "correct_answer": correction,
                "timestamp": datetime.now().isoformat(),
            }
            self._data["corrections"].append(pattern)

    def get_session(self, session_id: str) -> Optional[dict]:
        for s in self._data["sessions"]:
            if s["id"] == session_id:
                return s
        return None

    def find_similar_problems(self, problem_text: str, topic: str = None, top_k: int = 3) -> list:
        """
        Simple keyword-based similarity search in memory.
        Returns top_k most relevant past sessions.
        """
        from utils.llm_client import get_llm_response, parse_json_response

        if not self._data["sessions"]:
            return []

        # Build candidate list
        candidates = [
            s for s in self._data["sessions"]
            if s.get("feedback") == "correct"  # only use verified correct ones
        ]
        if topic:
            topic_candidates = [s for s in candidates if s.get("topic", "").lower() == topic.lower()]
            if topic_candidates:
                candidates = topic_candidates

        if not candidates:
            return []

        # Simple keyword overlap scoring
        query_words = set(problem_text.lower().split())
        scored = []
        for s in candidates:
            past_text = s.get("parsed_problem", {}).get("problem_text", "")
            past_words = set(past_text.lower().split())
            overlap = len(query_words & past_words) / max(len(query_words | past_words), 1)
            scored.append((overlap, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k] if _ > 0.1]

    def get_correction_patterns(self, topic: str = None) -> list:
        """Retrieve human corrections for a topic."""
        patterns = self._data["corrections"]
        if topic:
            patterns = [p for p in patterns if p.get("problem_topic", "").lower() == topic.lower()]
        return patterns[-10:]  # return last 10

    def get_stats(self) -> dict:
        sessions = self._data["sessions"]
        total = len(sessions)
        correct = sum(1 for s in sessions if s.get("feedback") == "correct")
        incorrect = sum(1 for s in sessions if s.get("feedback") == "incorrect")
        topics = {}
        for s in sessions:
            t = s.get("topic", "unknown")
            topics[t] = topics.get(t, 0) + 1
        return {
            "total_sessions": total,
            "correct_feedback": correct,
            "incorrect_feedback": incorrect,
            "no_feedback": total - correct - incorrect,
            "topics_distribution": topics,
            "total_corrections": len(self._data["corrections"]),
        }

    def get_recent_sessions(self, limit: int = 10) -> list:
        return self._data["sessions"][-limit:]


# Singleton
memory_store = MemoryStore()
