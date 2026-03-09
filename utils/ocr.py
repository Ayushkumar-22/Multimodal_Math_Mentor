"""
utils/ocr.py - OCR processing for image inputs
Supports EasyOCR (default) and Tesseract
"""
import os
import io
import numpy as np
from PIL import Image
from typing import Tuple
from config import config


def extract_text_from_image(image_bytes: bytes) -> Tuple[str, float]:
    """
    Extract text from image bytes.
    Returns (extracted_text, confidence_score 0-1)
    """
    engine = config.OCR_ENGINE.lower()

    if engine == "easyocr":
        return _easyocr_extract(image_bytes)
    elif engine == "tesseract":
        return _tesseract_extract(image_bytes)
    elif engine == "both":
        text1, conf1 = _easyocr_extract(image_bytes)
        text2, conf2 = _tesseract_extract(image_bytes)
        # Use whichever had higher confidence
        if conf1 >= conf2:
            return text1, conf1
        return text2, conf2
    else:
        return _easyocr_extract(image_bytes)


def _easyocr_extract(image_bytes: bytes) -> Tuple[str, float]:
    """EasyOCR extraction - better for mixed math/text content"""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        img_array = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
        results = reader.readtext(img_array)

        if not results:
            return "", 0.0

        # results: list of (bbox, text, confidence)
        texts = []
        confidences = []
        for (bbox, text, conf) in results:
            texts.append(text)
            confidences.append(conf)

        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return combined_text, avg_confidence

    except ImportError:
        return _tesseract_extract(image_bytes)
    except Exception as e:
        return f"OCR Error: {str(e)}", 0.0


def _tesseract_extract(image_bytes: bytes) -> Tuple[str, float]:
    """Tesseract OCR extraction"""
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Get text
        text = pytesseract.image_to_string(img, config='--psm 6')
        # Get confidence data
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [c for c in data['conf'] if c != -1]
        avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.5

        return text.strip(), avg_conf

    except Exception as e:
        return f"Tesseract Error: {str(e)}", 0.0


def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Preprocess image for better OCR:
    - Convert to grayscale
    - Increase contrast
    - Resize if too small
    """
    try:
        from PIL import ImageEnhance, ImageFilter
        img = Image.open(io.BytesIO(image_bytes)).convert('L')  # Grayscale

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Resize if small
        w, h = img.size
        if w < 800:
            scale = 800 / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Convert back to bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except Exception:
        return image_bytes
