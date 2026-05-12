"""
03_finetune.py
==============
Fine-tuning supervisionado (SFT) com LoRA/QLoRA usando HuggingFace Transformers + TRL.

Etapas:
  1. Carrega dados JSONL processados.
  2. Inicializa modelo base com quantização 4-bit (QLoRA) ou 16-bit (LoRA).
  3. Aplica adaptadores LoRA nos módulos de atenção e FFN.
  4. Executa SFT com SFTTrainer (TRL).
  5. Salva checkpoints e modelo final mesclado.
  6. Exporta relatório de treinamento.

Requisitos:
  pip install transformers trl peft bitsandbytes accelerate datasets torch

Uso:
  python 03_finetune.py [--config_override '{"learning_rate": 1e-4}']
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from config import BASE_DIR, LOGS_DIR, MODEL_DIR, REPORTS_DIR, FINETUNE_CONFIG, JSONL_TRAIN, JSONL_VALIDATION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "finetune.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verificação de dependências
# ---------------------------------------------------------------------------

def check_dependencies() -> bool:
    missing = []
    for pkg in ["transformers", "trl", "peft", "datasets", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(f"Dependências ausentes: {missing}")
        log.error("Execute: pip install transformers trl peft bitsandbytes accelerate datasets torch")
        return False
    return True


# ---------------------------------------------------------------------------
# Carregamento dos dados
# ---------------------------------------------------------------------------

def load_jsonl_dataset(train_path: Path, val_path: Path):
    """Carrega datasets JSONL no formato HuggingFace Dataset."""
    from datasets import load_dataset

    ds = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(val_path),
        },
    )
    log.info(f"Dataset carregado: train={len(ds['train'])} | validation={len(ds['validation'])}")
    return ds


# ---------------------------------------------------------------------------
# Formatação do chat (tokenização pelo template do modelo)
# ---------------------------------------------------------------------------

def format_messages_to_text(example: dict, tokenizer) -> dict:
    """
    Aplica o chat template do tokenizador para converter messages em texto.
    Gemma 2 não suporta role 'system' — mescla com a primeira mensagem user.
    """
    messages = example["messages"]

    # Mesclar system prompt com o primeiro user message (Gemma 2 não suporta system role)
    merged = []
    system_text = ""
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"].strip()
        else:
            merged.append(msg)

    if system_text and merged and merged[0]["role"] == "user":
        merged[0] = {
            "role": "user",
            "content": system_text + "\n\n" + merged[0]["content"],
        }

    text = tokenizer.apply_chat_template(
        merged,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Configuração do modelo com QLoRA / LoRA
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base_model = cfg["base_model"]
    use_qlora = cfg.get("use_qlora", True)

    log.info(f"Carregando tokenizador: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info(f"Carregando modelo base: {base_model} | QLoRA={use_qlora}")

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if _has_flash_attn() else "eager",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def _has_flash_attn() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Treinamento SFT
# ---------------------------------------------------------------------------

def run_training(cfg: dict, dataset, model, tokenizer) -> dict:
    from transformers import TrainingArguments
    from trl import SFTTrainer, SFTConfig
    import torch

    train_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=50,
        weight_decay=cfg["weight_decay"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        report_to="none",
        dataloader_num_workers=0,
        max_length=cfg["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    # Formata dataset
    log.info("Aplicando chat template aos dados de treino...")
    formatted_train = dataset["train"].map(
        lambda ex: format_messages_to_text(ex, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    formatted_val = dataset["validation"].map(
        lambda ex: format_messages_to_text(ex, tokenizer),
        remove_columns=dataset["validation"].column_names,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,
        args=train_args,
    )

    log.info("Iniciando treinamento SFT...")
    train_result = trainer.train()

    # Salvar modelo final
    log.info(f"Salvando modelo final em: {cfg['final_model_dir']}")
    trainer.save_model(cfg["final_model_dir"])
    tokenizer.save_pretrained(cfg["final_model_dir"])

    # Mesclar adaptadores LoRA no modelo base (para inferência eficiente)
    try:
        from peft import PeftModel
        log.info("Mesclando adaptadores LoRA no modelo base...")
        merged_dir = cfg["final_model_dir"] + "_merged"
        trainer.model.merge_and_unload().save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        log.info(f"Modelo mesclado salvo em: {merged_dir}")
    except Exception as e:
        log.warning(f"Não foi possível mesclar adaptadores: {e}")

    return {
        "train_loss": train_result.training_loss,
        "runtime_seconds": train_result.metrics.get("train_runtime", 0),
        "samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "global_step": train_result.global_step,
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning WomenHealth-LLM")
    parser.add_argument(
        "--config_override",
        type=str,
        default="{}",
        help='JSON string para sobrescrever configurações. Ex: \'{"learning_rate": 1e-4}\'',
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Apenas valida configurações sem executar treinamento.",
    )
    args = parser.parse_args()

    # Aplicar overrides
    cfg = dict(FINETUNE_CONFIG)
    try:
        overrides = json.loads(args.config_override)
        cfg.update(overrides)
    except json.JSONDecodeError as e:
        log.error(f"Erro no config_override JSON: {e}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("FINE-TUNING: WomenHealth-LLM")
    log.info(f"Modelo base: {cfg['base_model']}")
    log.info(f"QLoRA: {cfg['use_qlora']} | LoRA r={cfg['lora_r']} alpha={cfg['lora_alpha']}")
    log.info(f"Épocas: {cfg['num_train_epochs']} | LR: {cfg['learning_rate']}")
    log.info("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    # Verificar dados
    if not JSONL_TRAIN.exists():
        log.error(f"Dataset de treino não encontrado: {JSONL_TRAIN}")
        log.error("Execute primeiro: python 02_preprocess_data.py")
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN: configurações válidas. Treinamento não executado.")
        print(json.dumps(cfg, indent=2, default=str))
        return

    # Carregar dados
    dataset = load_jsonl_dataset(JSONL_TRAIN, JSONL_VALIDATION)

    # Carregar modelo
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Treinar
    start_time = datetime.now()
    metrics = run_training(cfg, dataset, model, tokenizer)
    end_time = datetime.now()

    # Relatório
    report = {
        "generated_at": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "config": {k: v for k, v in cfg.items() if k not in ("output_dir", "final_model_dir")},
        "metrics": metrics,
        "output": {
            "checkpoints_dir": cfg["output_dir"],
            "final_model_dir": cfg["final_model_dir"],
        },
        "next_steps": [
            "Executar 04_evaluate.py para avaliação clínica.",
            "Executar 05_clinical_tests.py para testes obrigatórios de segurança.",
            "Revisar exemplos com unsafe_flag=true nos dados de treino.",
        ],
    }

    report_path = REPORTS_DIR / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    log.info("TREINAMENTO CONCLUÍDO")
    log.info(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    log.info(f"  Duração: {(end_time - start_time).total_seconds() / 60:.1f} minutos")
    log.info(f"  Modelo final: {cfg['final_model_dir']}")
    log.info(f"  Relatório: {report_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
