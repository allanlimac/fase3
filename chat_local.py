"""
chat_local.py
=============
Chat interativo com o modelo WomenHealth-LLM fine-tunado localmente.
Não requer API — roda direto na sua GPU.

Uso:
    python chat_local.py
    python chat_local.py --model_dir model_output/womenhealth-llm-final_merged
    python chat_local.py --max_new_tokens 256
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configurações padrão
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = "model_output/womenhealth-llm-final_merged"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     WomenHealth-LLM  ·  Assistente de Saúde da Mulher       ║
║     Modelo local fine-tunado (Gemma 2 2B + QLoRA)           ║
║     Digite sua pergunta e pressione Enter.                   ║
║     'sair' ou Ctrl+C para encerrar.                         ║
╚══════════════════════════════════════════════════════════════╝
"""

AVISO = (
    "\n⚠️  AVISO: Este assistente fornece informações educativas sobre saúde da mulher.\n"
    "As respostas NÃO substituem avaliação médica profissional.\n"
)


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def merge_system_into_user(messages: list[dict]) -> list[dict]:
    """Gemma 2 não suporta role 'system'. Funde no primeiro user message."""
    if not messages or messages[0]["role"] != "system":
        return messages
    system_content = messages[0]["content"]
    remaining = list(messages[1:])
    for i, msg in enumerate(remaining):
        if msg["role"] == "user":
            remaining[i] = {
                "role": "user",
                "content": f"{system_content}\n\n{msg['content']}",
            }
            break
    return remaining


def build_messages(history: list[dict], user_input: str) -> list[dict]:
    """Monta a lista de mensagens com histórico + nova pergunta."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_input})
    return msgs


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    messages = merge_system_into_user(messages)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WomenHealth-LLM — chat local")
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Diretório do modelo mesclado (padrão: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Máximo de tokens gerados por resposta (padrão: 512)",
    )
    parser.add_argument(
        "--no_history",
        action="store_true",
        help="Desativa memória de conversa (cada pergunta é independente)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"[ERRO] Modelo não encontrado em: {model_path.resolve()}")
        print("Verifique se o fine-tuning foi concluído e o modelo foi salvo.")
        sys.exit(1)

    print(BANNER)
    print(f"Carregando modelo de: {model_path.resolve()}")
    print("Aguarde...\n")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Modelo carregado! Rodando em: {device}")
    print(AVISO)

    history: list[dict] = []

    try:
        while True:
            try:
                user_input = input("Você: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("sair", "exit", "quit"):
                print("Até logo!")
                break

            messages = build_messages(history, user_input)

            print("\nAssistente: ", end="", flush=True)
            try:
                response = generate(model, tokenizer, messages, args.max_new_tokens)
                print(response)
            except Exception as e:
                print(f"[ERRO ao gerar resposta: {e}]")
                continue

            print()

            if not args.no_history:
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})

                # Mantém no máximo os últimos 6 turnos (3 perguntas + 3 respostas)
                if len(history) > 12:
                    history = history[-12:]

    except KeyboardInterrupt:
        print("\nAté logo!")


if __name__ == "__main__":
    main()
