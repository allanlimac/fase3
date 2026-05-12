"""
chat.py
=======
Chat com a WomenHealth-LLM usando a API do Google Gemini.

Instalação:
    pip install google-genai python-dotenv

Uso:
    # Com chave no arquivo .env (recomendado):
    python chat.py

    # Passando a chave diretamente:
    python chat.py --api_key SUA_CHAVE_GEMINI

Obter chave gratuita em: https://aistudio.google.com/app/apikey
"""

import argparse
import os
import sys
from pathlib import Path

# Carrega variáveis do arquivo .env automaticamente
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv opcional; use --api_key se não instalado

# ---------------------------------------------------------------------------
# Prompt de sistema
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     WomenHealth-LLM  ·  Assistente de Saúde da Mulher       ║
║     Powered by Google Gemini                                 ║
║     Digite sua pergunta e pressione Enter.                   ║
║     'sair' para encerrar.                                    ║
╚══════════════════════════════════════════════════════════════╝
"""

AVISO = (
    "\n⚠️  AVISO: Este assistente fornece informações educativas sobre saúde "
    "da mulher.\nAs respostas NÃO substituem avaliação médica profissional.\n"
)


# ---------------------------------------------------------------------------
# Chat principal
# ---------------------------------------------------------------------------

def run_chat(api_key: str, model_name: str):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Instale o SDK do Gemini:")
        print("  pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print(BANNER)
    print(f"Modelo: {model_name}")
    print(AVISO)

    # Histórico acumulado da conversa
    history = []

    while True:
        try:
            user_input = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando. Cuide-se!")
            break

        if user_input.lower() in ("sair", "exit", "quit", "q"):
            print("Até logo! Cuide-se.")
            break

        if not user_input:
            continue

        # Adiciona turno do usuário ao histórico
        history.append(types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        ))

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=1024,
                ),
            )
            answer = response.text
        except Exception as e:
            print(f"\n[Erro Gemini] {e}\n")
            history.pop()  # remove turno sem resposta
            continue

        print(f"\nAssistente: {answer}\n")
        print("-" * 60)

        # Adiciona resposta do modelo ao histórico
        history.append(types.Content(
            role="model",
            parts=[types.Part(text=answer)],
        ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WomenHealth-LLM via Google Gemini"
    )
    parser.add_argument(
        "--api_key",
        default=os.getenv("GEMINI_API_KEY", ""),
        help="Chave de API do Google Gemini. Obtenha em: https://aistudio.google.com/app/apikey",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Modelo Gemini a usar (padrão: gemini-2.5-flash)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Chave de API não encontrada.")
        print("Cole sua chave no arquivo .env: GEMINI_API_KEY=sua_chave")
        print("Ou passe via argumento: python chat.py --api_key SUA_CHAVE")
        sys.exit(1)

    run_chat(api_key=args.api_key, model_name=args.model)


if __name__ == "__main__":
    main()
