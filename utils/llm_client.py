# """
# utils/llm_client.py - Unified LLM client supporting Anthropic and OpenAI
# """
# import json
# from typing import Optional
# from config import config


# def get_llm_response(
#     prompt: str,
#     system: str = "",
#     temperature: float = 0.2,
#     max_tokens: int = 4096,
#     json_mode: bool = False,
# ) -> str:
#     """
#     Unified LLM call. Routes to Anthropic or OpenAI based on config.
#     Returns raw string response.
#     """
#     if config.LLM_PROVIDER == "anthropic":
#         return _anthropic_call(prompt, system, temperature, max_tokens, json_mode)
#     else:
#         return _openai_call(prompt, system, temperature, max_tokens, json_mode)


# def _anthropic_call(prompt, system, temperature, max_tokens, json_mode) -> str:
#     try:
#         import anthropic
#         client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

#         system_msg = system
#         if json_mode:
#             system_msg += "\n\nYou MUST respond with valid JSON only. No markdown, no explanation outside JSON."

#         messages = [{"role": "user", "content": prompt}]

#         response = client.messages.create(
#             model=config.LLM_MODEL,
#             max_tokens=max_tokens,
#             system=system_msg if system_msg else "You are a helpful math tutor.",
#             messages=messages,
#             temperature=temperature,
#         )
#         return response.content[0].text
#     except Exception as e:
#         return f"ERROR: LLM call failed: {str(e)}"


# def _openai_call(prompt, system, temperature, max_tokens, json_mode) -> str:
#     try:
#         from openai import OpenAI
#         client = OpenAI(api_key=config.OPENAI_API_KEY)

#         messages = []
#         if system:
#             messages.append({"role": "system", "content": system})
#         messages.append({"role": "user", "content": prompt})

#         kwargs = {
#             "model": config.LLM_MODEL,
#             "messages": messages,
#             "temperature": temperature,
#             "max_tokens": max_tokens,
#         }
#         if json_mode:
#             kwargs["response_format"] = {"type": "json_object"}

#         response = client.chat.completions.create(**kwargs)
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"ERROR: LLM call failed: {str(e)}"


# def parse_json_response(text: str) -> dict:
#     """Safely parse JSON from LLM response, stripping markdown fences."""
#     text = text.strip()
#     # Strip markdown code fences
#     for fence in ["```json", "```JSON", "```"]:
#         if text.startswith(fence):
#             text = text[len(fence):]
#             break
#     if text.endswith("```"):
#         text = text[:-3]
#     text = text.strip()
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         # Try to extract JSON object from surrounding text
#         import re
#         match = re.search(r'\{.*\}', text, re.DOTALL)
#         if match:
#             try:
#                 return json.loads(match.group())
#             except:
#                 pass
#         return {"error": "Failed to parse JSON", "raw": text}

import json
import requests
from typing import Optional
from config import config


def get_llm_response(
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    if config.LLM_PROVIDER == "anthropic":
        return _anthropic_call(prompt, system, temperature, max_tokens, json_mode)
    elif config.LLM_PROVIDER == "huggingface":
        return _huggingface_call(prompt, system, temperature, max_tokens, json_mode)
    else:
        return _openai_call(prompt, system, temperature, max_tokens, json_mode)


def _anthropic_call(prompt, system, temperature, max_tokens, json_mode) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        system_msg = system
        if json_mode:
            system_msg += "\n\nYou MUST respond with valid JSON only. No markdown, no explanation outside JSON."
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=max_tokens,
            system=system_msg if system_msg else "You are a helpful math tutor.",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.content[0].text
    except Exception as e:
        return f"ERROR: LLM call failed: {str(e)}"


def _openai_call(prompt, system, temperature, max_tokens, json_mode) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: LLM call failed: {str(e)}"


# def _huggingface_call(prompt, system, temperature, max_tokens, json_mode) -> str:
#     try:
#         full_prompt = ""
#         if system:
#             full_prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
#         else:
#             full_prompt = f"<s>[INST] {prompt} [/INST]"

#         if json_mode:
#             full_prompt += "\nRespond with valid JSON only."

#         headers = {"Authorization": f"Bearer {config.HF_API_KEY}"}
#         payload = {
#             "inputs": full_prompt,
#             "parameters": {
#                 "max_new_tokens": max_tokens,
#                 "temperature": temperature,
#                 "return_full_text": False,
#             }
#         }

#         url = f"https://api-inference.huggingface.co/models/{config.LLM_MODEL}"
#         response = requests.post(url, headers=headers, json=payload, timeout=60)
#         response.raise_for_status()

#         data = response.json()
#         if isinstance(data, list):
#             return data[0].get("generated_text", "")
#         return str(data)

#     except Exception as e:
#         return f"ERROR: HuggingFace call failed: {str(e)}"
def _huggingface_call(prompt, system, temperature, max_tokens, json_mode) -> str:
    try:
        api_key = config.HF_API_KEY

        if not api_key or not api_key.startswith("hf_"):
            return "ERROR: HF_API_KEY not set correctly in .env file. Must start with hf_"

        model = config.LLM_MODEL

        # Build prompt based on model type
        if "zephyr" in model.lower():
            full_prompt = f"<|system|>\n{system}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"

        elif "tinyllama" in model.lower():
            full_prompt = f"<|system|>\n{system}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"

        elif "phi" in model.lower():
            full_prompt = f"Instruct: {system}\n{prompt}\nOutput:"

        elif "flan" in model.lower():
            full_prompt = f"{system}\n\n{prompt}" if system else prompt

        else:
            # Generic instruct format
            full_prompt = f"[INST] {system}\n\n{prompt} [/INST]" if system else f"[INST] {prompt} [/INST]"

        if json_mode:
            full_prompt += "\nRespond with valid JSON only. No extra text outside the JSON."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": min(max_tokens, 1500),
                "temperature": max(temperature, 0.01),
                "return_full_text": False,
                "do_sample": True,
            },
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }

        url = f"https://api-inference.huggingface.co/models/{model}"
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code == 410:
            return "ERROR: This model is no longer available on HuggingFace free API. Change LLM_MODEL in your .env file to: HuggingFaceH4/zephyr-7b-beta"

        if response.status_code == 503:
            return "ERROR: Model is loading. Wait 30 seconds and try again."

        if response.status_code == 429:
            return "ERROR: Rate limit hit. Wait 1 minute and try again."

        response.raise_for_status()

        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "").strip()
        elif isinstance(data, dict) and "error" in data:
            return f"ERROR: {data['error']}"

        return str(data)

    except requests.exceptions.Timeout:
        return "ERROR: Request timed out. Try again."
    except Exception as e:
        return f"ERROR: HuggingFace call failed: {str(e)}"


def parse_json_response(text: str) -> dict:
    text = text.strip()
    for fence in ["```json", "```JSON", "```"]:
        if text.startswith(fence):
            text = text[len(fence):]
            break
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {"error": "Failed to parse JSON", "raw": text}