"""
01_audit_data.py
================
Auditoria completa de todos os datasets Excel antes do pré-processamento.

Verifica:
- Duplicidades
- Exemplos vazios ou corrompidos
- Idioma predominante
- Respostas potencialmente inseguras
- Distribuição por domínio clínico
- Nível de risco estimado
- Qualidade das pontuações (domestic violence)

Gera relatório em: reports/data_audit_report.json
"""

import json
import re
import ast
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from config import (
    BASE_DIR, REPORTS_DIR, EXCEL_FILES,
    EMERGENCY_KEYWORDS, HIGH_RISK_KEYWORDS, MODERATE_RISK_KEYWORDS,
    DV_MIN_CORRECTNESS, DV_MIN_HELPFULNESS, DV_MIN_COHERENCE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "audit.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

UNSAFE_PATTERNS = [
    r"\btome\s+\d+\s*(mg|mcg|ml|comprimido)\b",
    r"\bprescrev[oaei]\b",
    r"\buse\s+\d+\s*(mg|comprimido)\b",
    r"\bdosagem\s+de\s+\d+",
    r"\bnão\s+precisa\s+ir\s+ao\s+médico\b",
    r"\bnão\s+é\s+necessário\s+consultar\b",
    r"\bdiagnóstico\s+definitivo\b",
    r"\bvocê\s+tem\s+certamente\b",
    r"\bcom\s+certeza\s+é\s+(um|uma)\b",
]

UNSAFE_RE = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


def estimate_risk(text: str) -> str:
    """Estima nível de risco clínico baseado em palavras-chave."""
    tl = text.lower()
    if any(kw in tl for kw in EMERGENCY_KEYWORDS):
        return "emergência"
    if any(kw in tl for kw in HIGH_RISK_KEYWORDS):
        return "alto"
    if any(kw in tl for kw in MODERATE_RISK_KEYWORDS):
        return "moderado"
    return "baixo"


def has_unsafe_content(text: str) -> bool:
    """Verifica padrões de conteúdo clinicamente inseguro."""
    return any(p.search(text) for p in UNSAFE_RE)


def text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def is_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float):
        import math
        return math.isnan(val)
    return str(val).strip() == ""


def classify_domain_obstetric(text: str) -> str:
    tl = text.lower()
    if any(w in tl for w in ["neonato", "recém-nascido", "neonatal", "infant", "newborn"]):
        return "neonatologia"
    if any(w in tl for w in ["puerpério", "pós-parto", "postpartum", "lochia", "lóquio"]):
        return "obstetrícia"
    if any(w in tl for w in ["amamentação", "breastfeed", "amament", "lactação"]):
        return "amamentação"
    if any(w in tl for w in ["parto", "labor", "delivery", "cesárea", "cesarean"]):
        return "obstetrícia"
    if any(w in tl for w in ["gravidez", "gestação", "prenatal", "pré-natal", "pregnancy"]):
        return "obstetrícia"
    return "obstetrícia"


# ---------------------------------------------------------------------------
# Auditores por dataset
# ---------------------------------------------------------------------------

def audit_musicotherapy(path: Path) -> dict:
    log.info(f"Auditando: {path.name}")
    results = {}
    for sheet in ["train", "validation"]:
        df = pd.read_excel(path, sheet_name=sheet)
        total = len(df)
        empty = df["text"].apply(is_empty).sum()
        hashes = df["text"].dropna().apply(lambda x: text_hash(str(x)))
        duplicates = int(hashes.duplicated().sum())
        unsafe_count = 0
        risk_dist = {"baixo": 0, "moderado": 0, "alto": 0, "emergência": 0}

        for _, row in df.iterrows():
            txt = str(row.get("text", ""))
            if has_unsafe_content(txt):
                unsafe_count += 1
            risk = estimate_risk(txt)
            risk_dist[risk] += 1

        results[sheet] = {
            "total_rows": total,
            "empty_rows": int(empty),
            "duplicate_rows": duplicates,
            "unsafe_content_rows": unsafe_count,
            "risk_distribution": risk_dist,
            "usable_rows": total - int(empty) - duplicates,
        }
        log.info(f"  [{sheet}] Total={total} | Vazios={empty} | Duplicados={duplicates} | Inseguros={unsafe_count}")
    return results


def audit_women_health(path: Path) -> dict:
    log.info(f"Auditando: {path.name}")
    df = pd.read_excel(path, sheet_name="train")
    total = len(df)
    empty = df["conversations"].apply(is_empty).sum()
    parse_errors = 0
    unsafe_count = 0
    risk_dist = {"baixo": 0, "moderado": 0, "alto": 0, "emergência": 0}
    multi_turn = 0

    for _, row in df.iterrows():
        raw = row.get("conversations", "")
        if is_empty(raw):
            continue
        try:
            fixed = re.sub(r'\}\s*\n\s*\{', '}, {', str(raw))
            convs = ast.literal_eval(fixed)
            if not isinstance(convs, list):
                parse_errors += 1
                continue
            if len(convs) > 2:
                multi_turn += 1
            full_text = " ".join(c.get("content", "") for c in convs)
            if has_unsafe_content(full_text):
                unsafe_count += 1
            risk = estimate_risk(full_text)
            risk_dist[risk] += 1
        except Exception:
            parse_errors += 1

    result = {
        "train": {
            "total_rows": total,
            "empty_rows": int(empty),
            "parse_errors": parse_errors,
            "multi_turn_conversations": multi_turn,
            "unsafe_content_rows": unsafe_count,
            "risk_distribution": risk_dist,
            "usable_rows": total - int(empty) - parse_errors,
        }
    }
    log.info(f"  [train] Total={total} | Vazios={int(empty)} | Erros Parse={parse_errors} | Inseguros={unsafe_count}")
    return result


def audit_domestic_violence(path: Path) -> dict:
    log.info(f"Auditando: {path.name}")
    df = pd.read_excel(path, sheet_name="train")
    total = len(df)

    required_cols = ["response", "helpfulness", "correctness", "coherence", "question"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        log.warning(f"  Colunas faltando: {missing_cols}")

    empty_q = df["question"].apply(is_empty).sum() if "question" in df.columns else 0
    empty_r = df["response"].apply(is_empty).sum() if "response" in df.columns else 0

    # Filtro de qualidade
    if all(c in df.columns for c in ["correctness", "helpfulness", "coherence"]):
        high_quality = df[
            (df["correctness"] >= DV_MIN_CORRECTNESS) &
            (df["helpfulness"] >= DV_MIN_HELPFULNESS) &
            (df["coherence"] >= DV_MIN_COHERENCE)
        ]
        hq_count = len(high_quality)
    else:
        hq_count = 0

    score_stats = {}
    for col in ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]:
        if col in df.columns:
            score_stats[col] = {
                "mean": round(float(df[col].mean()), 3),
                "min": round(float(df[col].min()), 3),
                "max": round(float(df[col].max()), 3),
            }

    unsafe_count = 0
    for _, row in df.iterrows():
        txt = str(row.get("response", "")) + " " + str(row.get("question", ""))
        if has_unsafe_content(txt):
            unsafe_count += 1

    result = {
        "train": {
            "total_rows": total,
            "empty_questions": int(empty_q),
            "empty_responses": int(empty_r),
            "high_quality_rows": hq_count,
            "unsafe_content_rows": unsafe_count,
            "score_statistics": score_stats,
            "filter_thresholds": {
                "correctness": DV_MIN_CORRECTNESS,
                "helpfulness": DV_MIN_HELPFULNESS,
                "coherence": DV_MIN_COHERENCE,
            },
            "usable_rows": hq_count,
        }
    }
    log.info(f"  [train] Total={total} | Alta qualidade={hq_count} | Inseguros={unsafe_count}")
    return result


def audit_menstrual_health(path: Path) -> dict:
    log.info(f"Auditando: {path.name}")
    results = {}
    for sheet in ["train", "test"]:
        df = pd.read_excel(path, sheet_name=sheet)
        inst_col = "instruction (string)"
        out_col = "output (string)"

        total = len(df)
        empty_inst = df[inst_col].apply(is_empty).sum() if inst_col in df.columns else total
        empty_out = df[out_col].apply(is_empty).sum() if out_col in df.columns else total

        hashes = (
            (df[inst_col].fillna("") + df[out_col].fillna(""))
            .apply(lambda x: text_hash(str(x)))
        )
        duplicates = int(hashes.duplicated().sum())
        unsafe_count = 0
        risk_dist = {"baixo": 0, "moderado": 0, "alto": 0, "emergência": 0}

        for _, row in df.iterrows():
            txt = str(row.get(inst_col, "")) + " " + str(row.get(out_col, ""))
            if has_unsafe_content(txt):
                unsafe_count += 1
            risk = estimate_risk(txt)
            risk_dist[risk] += 1

        results[sheet] = {
            "total_rows": total,
            "empty_instructions": int(empty_inst),
            "empty_outputs": int(empty_out),
            "duplicate_rows": duplicates,
            "unsafe_content_rows": unsafe_count,
            "risk_distribution": risk_dist,
            "usable_rows": total - int(empty_inst) - int(empty_out) - duplicates,
        }
        log.info(f"  [{sheet}] Total={total} | Vazios inst={int(empty_inst)} | Vazios out={int(empty_out)} | Duplicados={duplicates}")
    return results


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def run_audit() -> dict:
    log.info("=" * 60)
    log.info("INICIANDO AUDITORIA DOS DATASETS")
    log.info("=" * 60)

    report = {
        "generated_at": datetime.now().isoformat(),
        "datasets": {},
        "summary": {},
    }

    auditors = {
        "musicotherapy_obstetric": audit_musicotherapy,
        "women_health_mini": audit_women_health,
        "domestic_violence": audit_domestic_violence,
        "menstrual_health": audit_menstrual_health,
    }

    total_usable = 0
    for key, auditor_fn in auditors.items():
        path = EXCEL_FILES[key]
        if not path.exists():
            log.error(f"Arquivo não encontrado: {path}")
            report["datasets"][key] = {"error": "arquivo não encontrado"}
            continue
        dataset_report = auditor_fn(path)
        report["datasets"][key] = dataset_report

        # Somar usable_rows de todas as abas
        for sheet_data in dataset_report.values():
            if isinstance(sheet_data, dict):
                total_usable += sheet_data.get("usable_rows", 0)

    report["summary"] = {
        "total_usable_examples": total_usable,
        "audit_status": "concluída",
        "next_step": "Executar 02_preprocess_data.py para converter para JSONL",
        "recommendations": [
            "domestic-violence-dataset-prep.xlsx tem poucos exemplos (14 total, 7 de alta qualidade). "
            "Considere buscar datasets adicionais de violência doméstica.",
            "women-health-mini.xlsx tem conversas em inglês. O pré-processamento incluirá "
            "campo de idioma para filtragem ou tradução futura.",
            "musicotherapy_obstetric_dataset.xlsx contém tokens LLaMA que serão removidos no pré-processamento.",
            "Todos os exemplos com conteúdo inseguro serão revisados e rotulados no pré-processamento.",
        ],
    }

    report_path = REPORTS_DIR / "data_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    log.info(f"AUDITORIA CONCLUÍDA")
    log.info(f"Total de exemplos utilizáveis estimados: {total_usable}")
    log.info(f"Relatório salvo em: {report_path}")
    log.info("=" * 60)

    return report


if __name__ == "__main__":
    report = run_audit()
    print("\n=== RESUMO DA AUDITORIA ===")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
