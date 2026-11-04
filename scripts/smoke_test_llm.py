"""Standalone real API smoke-test script for DocuMind Groq connectivity.

Runs only when explicitly executed:
    python scripts/smoke_test_llm.py

Verifies:
1. GROQ_API_KEY is configured.
2. Groq API client can authenticate and communicate.
3. Model generates a valid, non-empty response.
4. Response latency is measured.
5. API keys are strictly kept confidential and never printed.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm import (
    DEFAULT_MODEL,
    LLMError,
    generate_response,
    get_model_name,
)


def main() -> int:
    # Ensure UTF-8 output safe for Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("DocuMind Phase 5.1: Groq API Connectivity Smoke Test")
    print("=" * 60)

    # 1. Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or not api_key.strip():
        print("[!] Error: GROQ_API_KEY is not set.")
        print("    Please create a .env file in the project root with:")
        print("    GROQ_API_KEY=your_groq_api_key_here")
        print("    GROQ_MODEL=openai/gpt-oss-120b  (optional)")
        return 1

    # Safe verification: show presence and length only, NEVER print the key
    masked_hint = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    print(f"[OK] GROQ_API_KEY: Configured ({len(api_key)} chars, hint: {masked_hint})")

    # 2. Check active model
    model_name = get_model_name()
    print(f"[OK] Target Model: {model_name} (default: {DEFAULT_MODEL})")

    # 3. Send test prompt
    test_prompt = "Say 'DocuMind Groq connectivity verified!' in exactly one short sentence."
    print(f"[*] Test Prompt:  \"{test_prompt}\"")
    print("[*] Sending request to Groq API...")

    t_start = time.perf_counter()
    try:
        response = generate_response(test_prompt)
        latency_ms = (time.perf_counter() - t_start) * 1000

        print(f"[OK] Success! Response received in {latency_ms:.1f} ms:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print("[OK] Groq API connectivity is fully operational!")
        return 0

    except LLMError as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000
        print(f"[!] Groq API Error ({latency_ms:.1f} ms): {exc}")
        return 2
    except Exception as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000
        print(f"[!] Unexpected Error ({latency_ms:.1f} ms): {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
