# """
# config.py - Centralized configuration management
# """
# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Config:
#     # LLM
#     ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
#     LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
#     LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-4-5")

#     # Embeddings
#     EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
#     EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

#     # Vector Store
#     VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")
#     CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

#     # Memory
#     MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "./data/memory.json")

#     # OCR
#     OCR_ENGINE = os.getenv("OCR_ENGINE", "easyocr")
#     TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

#     # ASR
#     WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

#     # App
#     APP_TITLE = os.getenv("APP_TITLE", "Math Mentor AI")
#     DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

#     # HITL Thresholds
#     OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6"))
#     ASR_CONFIDENCE_THRESHOLD = float(os.getenv("ASR_CONFIDENCE_THRESHOLD", "0.7"))
#     VERIFIER_CONFIDENCE_THRESHOLD = float(os.getenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.75"))

# config = Config()


"""
config.py - Centralized configuration management
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip()
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o").strip()

    # Embeddings
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Vector Store
    VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

    # Memory
    MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "./data/memory.json")

    # OCR
    OCR_ENGINE = os.getenv("OCR_ENGINE", "easyocr")
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

    # ASR
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

    # App
    APP_TITLE = os.getenv("APP_TITLE", "Math Mentor AI")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # HITL Thresholds
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6"))
    ASR_CONFIDENCE_THRESHOLD = float(os.getenv("ASR_CONFIDENCE_THRESHOLD", "0.7"))
    VERIFIER_CONFIDENCE_THRESHOLD = float(os.getenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.75"))

config = Config()