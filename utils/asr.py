"""
utils/asr.py - Speech-to-Text processing using OpenAI Whisper
"""
import os
import io
import tempfile
from typing import Tuple
from config import config


def transcribe_audio(audio_bytes: bytes, file_format: str = "wav") -> Tuple[str, float]:
    """
    Transcribe audio bytes to text using Whisper.
    Returns (transcript, confidence_score 0-1)
    """
    try:
        import whisper
        import numpy as np

        # Save to temp file (Whisper needs file path)
        suffix = f".{file_format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            model = whisper.load_model(config.WHISPER_MODEL)
            result = model.transcribe(tmp_path, language="en")

            transcript = result["text"].strip()
            # Whisper doesn't give per-utterance confidence directly
            # Use avg log-prob as proxy (range roughly -1 to 0, higher is better)
            avg_logprob = result.get("segments", [{}])[0].get("avg_logprob", -0.5) if result.get("segments") else -0.5
            # Map avg_logprob to 0-1: typically -1.0 to 0.0
            confidence = max(0.0, min(1.0, 1.0 + avg_logprob))

            return transcript, confidence
        finally:
            os.unlink(tmp_path)

    except ImportError:
        return "Whisper not installed. Please run: pip install openai-whisper", 0.0
    except Exception as e:
        return f"ASR Error: {str(e)}", 0.0


def normalize_math_transcript(transcript: str) -> str:
    """
    Normalize math-specific speech patterns:
    e.g., "square root of 16" -> "√16"
         "x raised to the power 2" -> "x^2"
         "pi" -> "π"
    """
    replacements = [
        ("square root of", "√"),
        ("square root", "√"),
        ("raised to the power of", "^"),
        ("raised to the power", "^"),
        ("to the power of", "^"),
        ("to the power", "^"),
        ("divided by", "/"),
        ("multiplied by", "*"),
        ("plus or minus", "±"),
        ("plus", "+"),
        ("minus", "-"),
        ("times", "*"),
        ("equals", "="),
        ("is equal to", "="),
        ("greater than or equal to", "≥"),
        ("less than or equal to", "≤"),
        ("greater than", ">"),
        ("less than", "<"),
        ("infinity", "∞"),
        ("alpha", "α"),
        ("beta", "β"),
        ("theta", "θ"),
        ("pi", "π"),
        ("sigma", "σ"),
        ("delta", "Δ"),
        ("integral of", "∫"),
        ("sum of", "Σ"),
        ("log base", "log_"),
        ("natural log", "ln"),
        ("absolute value of", "|"),
    ]

    text = transcript.lower()
    for phrase, symbol in replacements:
        text = text.replace(phrase, symbol)
    return text
