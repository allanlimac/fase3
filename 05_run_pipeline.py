"""
05_run_pipeline.py
==================
Orquestrador completo do pipeline WomenHealth-LLM.

Executa na ordem correta:
  1. Auditoria dos dados (01_audit_data.py)
  2. Pré-processamento e geração de JSONL (02_preprocess_data.py)
  3. Fine-tuning (03_finetune.py) — opcional, requer GPU
  4. Avaliação clínica (04_evaluate.py)

Uso:
  # Executar apenas auditoria e pré-processamento:
  python 05_run_pipeline.py --skip_training

  # Pipeline completo (requer GPU e dependências de treino):
  python 05_run_pipeline.py

  # Executar apenas avaliação com modelo existente:
  python 05_run_pipeline.py --only_eval --model_dir model_output/womenhealth-llm-final
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import BASE_DIR, LOGS_DIR, REPORTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def run_step(label: str, script: str, extra_args: list[str] = None) -> bool:
    """Executa um script Python e retorna True se bem-sucedido."""
    cmd = [sys.executable, script] + (extra_args or [])
    log.info(f"\n{'=' * 60}")
    log.info(f"ETAPA: {label}")
    log.info(f"Comando: {' '.join(cmd)}")
    log.info(f"{'=' * 60}")

    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        log.error(f"FALHA na etapa: {label} (código {result.returncode})")
        return False
    log.info(f"OK: {label}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Pipeline WomenHealth-LLM")
    parser.add_argument("--skip_training", action="store_true", help="Pula etapa de fine-tuning")
    parser.add_argument("--only_eval", action="store_true", help="Executa apenas avaliação")
    parser.add_argument("--model_dir", type=str, help="Modelo para avaliação")
    parser.add_argument("--simulate_eval", action="store_true", default=True,
                        help="Usa simulação na avaliação (padrão: True)")
    args = parser.parse_args()

    start = datetime.now()
    steps_ok = []
    steps_fail = []

    log.info("=" * 60)
    log.info("INICIANDO PIPELINE WOMENHEALTH-LLM")
    log.info(f"Início: {start.isoformat()}")
    log.info("=" * 60)

    if not args.only_eval:
        # Etapa 1: Auditoria
        ok = run_step("Auditoria dos Dados", "01_audit_data.py")
        (steps_ok if ok else steps_fail).append("auditoria")
        if not ok:
            log.warning("Auditoria falhou mas continuando pré-processamento...")

        # Etapa 2: Pré-processamento
        ok = run_step("Pré-processamento e JSONL", "02_preprocess_data.py")
        (steps_ok if ok else steps_fail).append("preprocessamento")
        if not ok:
            log.error("Pré-processamento falhou. Abortando pipeline.")
            sys.exit(1)

        # Etapa 3: Fine-tuning (opcional)
        if not args.skip_training:
            extra = []
            if args.model_dir:
                extra = [f'--config_override={{"output_dir":"{args.model_dir}"}}']
            ok = run_step("Fine-Tuning SFT com LoRA/QLoRA", "03_finetune.py", extra)
            (steps_ok if ok else steps_fail).append("finetune")
            if not ok:
                log.warning("Fine-tuning falhou. Continuando com avaliação em modo simulação.")
                args.simulate_eval = True
        else:
            log.info("Fine-tuning ignorado (--skip_training)")
            steps_ok.append("finetune (ignorado)")

    # Etapa 4: Avaliação
    eval_args = []
    if args.simulate_eval or not args.model_dir:
        eval_args.append("--simulate")
    elif args.model_dir:
        eval_args.extend(["--model_dir", args.model_dir])

    ok = run_step("Avaliação Clínica", "04_evaluate.py", eval_args)
    (steps_ok if ok else steps_fail).append("avaliação")

    end = datetime.now()
    duration = (end - start).total_seconds()

    # Relatório do pipeline
    pipeline_report = {
        "generated_at": end.isoformat(),
        "duration_seconds": duration,
        "steps_successful": steps_ok,
        "steps_failed": steps_fail,
        "overall_status": "SUCESSO" if not steps_fail else "PARCIAL" if steps_ok else "FALHA",
        "next_steps": [
            "Revisar reports/ para relatórios detalhados de cada etapa.",
            "Validar exemplos com unsafe_flag=true no dataset de treino.",
            "Obter revisão de especialista clínico antes do uso em produção.",
            "Consultar SAFETY_GUIDE.md antes de implantar o modelo.",
        ],
    }

    report_path = REPORTS_DIR / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_report, f, ensure_ascii=False, indent=2)

    log.info("\n" + "=" * 60)
    log.info("PIPELINE CONCLUÍDO")
    log.info(f"Status: {pipeline_report['overall_status']}")
    log.info(f"Duração: {duration:.1f}s ({duration/60:.1f} min)")
    log.info(f"Etapas OK: {steps_ok}")
    log.info(f"Etapas com falha: {steps_fail}")
    log.info(f"Relatório: {report_path}")
    log.info("=" * 60)

    sys.exit(0 if not steps_fail else 1)


if __name__ == "__main__":
    main()
