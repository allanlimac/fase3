"""
02_preprocess_data.py
=====================
Pré-processamento completo de todos os datasets Excel.

Converte para formato JSONL único com estrutura:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "domain": "...",
    "source_file": "...",
    "risk_level": "...",
    "requires_referral": true/false,
    "language": "pt|en|mixed",
    "unsafe_flag": false
  }
}

Gera:
  - data/processed/train.jsonl
  - data/processed/validation.jsonl
  - data/processed/test.jsonl
  - reports/preprocessing_report.json
"""

import ast
import json
import logging
import random
import re
import hashlib
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    BASE_DIR, PROCESSED_DIR, REPORTS_DIR, EXCEL_FILES,
    SYSTEM_PROMPT,
    EMERGENCY_KEYWORDS, HIGH_RISK_KEYWORDS, MODERATE_RISK_KEYWORDS,
    DV_MIN_CORRECTNESS, DV_MIN_HELPFULNESS, DV_MIN_COHERENCE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "preprocess.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

random.seed(42)

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

# Tokens de template LLaMA a remover
LLAMA_TOKENS_RE = re.compile(r"</?s>|\[/?INST\]", re.IGNORECASE)


def remove_llama_tokens(text: str) -> str:
    return LLAMA_TOKENS_RE.sub("", text).strip()


def is_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return str(val).strip() == ""


def text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def estimate_risk(text: str) -> str:
    tl = text.lower()
    if any(kw in tl for kw in EMERGENCY_KEYWORDS):
        return "emergência"
    if any(kw in tl for kw in HIGH_RISK_KEYWORDS):
        return "alto"
    if any(kw in tl for kw in MODERATE_RISK_KEYWORDS):
        return "moderado"
    return "baixo"


def requires_referral(text: str, risk: str) -> bool:
    return risk in ("alto", "emergência")


UNSAFE_PATTERNS_RE = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\btome\s+\d+\s*(mg|mcg|ml|comprimido)\b",
        r"\buse\s+\d+\s*(mg|comprimido)\b",
        r"\bdosagem\s+de\s+\d+",
        r"\bnão\s+precisa\s+ir\s+ao\s+médico\b",
        r"\bnão\s+é\s+necessário\s+consultar\b",
    ]
]


def has_unsafe_content(text: str) -> bool:
    return any(p.search(text) for p in UNSAFE_PATTERNS_RE)


def detect_language(text: str) -> str:
    """Detecção simples por presença de stopwords."""
    pt_words = {"você", "não", "que", "para", "com", "uma", "por", "seu", "sua", "são", "mais", "como"}
    en_words = {"the", "you", "that", "with", "this", "have", "from", "they", "will", "your"}
    tokens = set(text.lower().split())
    pt_score = len(tokens & pt_words)
    en_score = len(tokens & en_words)
    if pt_score > en_score:
        return "pt"
    if en_score > pt_score:
        return "en"
    return "mixed"


def make_example(
    user_msg: str,
    assistant_msg: str,
    domain: str,
    source_file: str,
    risk_level: Optional[str] = None,
    language: str = "pt",
) -> dict:
    """Monta um exemplo no formato JSONL padrão."""
    full_text = user_msg + " " + assistant_msg
    rl = risk_level or estimate_risk(full_text)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": assistant_msg.strip()},
        ],
        "metadata": {
            "domain": domain,
            "source_file": source_file,
            "risk_level": rl,
            "requires_referral": requires_referral(full_text, rl),
            "language": language,
            "unsafe_flag": has_unsafe_content(assistant_msg),
        },
    }


# ---------------------------------------------------------------------------
# Processadores por dataset
# ---------------------------------------------------------------------------

def process_musicotherapy(path: Path, split: str) -> list[dict]:
    """
    Extrai pares [INST]/[/INST] do campo text.
    Remove tokens LLaMA. Separa múltiplas interações.
    """
    df = pd.read_excel(path, sheet_name=split)
    examples = []
    seen = set()

    for _, row in df.iterrows():
        raw = str(row.get("text", ""))
        if is_empty(raw):
            continue

        # Divide em blocos instrução→resposta
        # Padrão: [INST] ... [/INST] ... (próximo [INST] ou fim)
        blocks = re.split(r"\[INST\]", raw, flags=re.IGNORECASE)
        for block in blocks:
            if not block.strip():
                continue
            parts = re.split(r"\[/INST\]", block, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) < 2:
                continue
            user_raw = remove_llama_tokens(parts[0]).strip()
            assistant_raw = remove_llama_tokens(parts[1]).strip()

            if not user_raw or not assistant_raw:
                continue
            if len(user_raw) < 10 or len(assistant_raw) < 20:
                continue

            h = text_hash(user_raw + assistant_raw)
            if h in seen:
                continue
            seen.add(h)

            domain = classify_domain_obstetric(user_raw + " " + assistant_raw)
            lang = detect_language(user_raw)
            examples.append(make_example(user_raw, assistant_raw, domain, path.name, language=lang))

    log.info(f"  musicotherapy [{split}]: {len(examples)} exemplos extraídos")
    return examples


def classify_domain_obstetric(text: str) -> str:
    tl = text.lower()
    if any(w in tl for w in ["neonato", "recém-nascido", "neonatal", "infant", "newborn"]):
        return "neonatologia"
    if any(w in tl for w in ["amamentação", "breastfeed", "lactação", "nursing"]):
        return "amamentação"
    if any(w in tl for w in ["puerpério", "pós-parto", "postpartum", "lochia"]):
        return "obstetrícia"
    if any(w in tl for w in ["parto", "labor", "delivery", "cesárea", "cesarean"]):
        return "obstetrícia"
    if any(w in tl for w in ["gravidez", "gestação", "prenatal", "pré-natal", "pregnancy"]):
        return "obstetrícia"
    return "obstetrícia"


def process_women_health(path: Path) -> list[dict]:
    """
    Converte lista de conversas para estrutura messages.
    Mantém role=user e role=assistant.
    Remove exemplos com flag insegura.
    """
    df = pd.read_excel(path, sheet_name="train")
    examples = []
    seen = set()
    skipped_unsafe = 0
    skipped_parse = 0

    for _, row in df.iterrows():
        raw = row.get("conversations", "")
        if is_empty(raw):
            continue
        try:
            fixed_raw = re.sub(r'\}\s*\n\s*\{', '}, {', str(raw))
            convs = ast.literal_eval(fixed_raw)
            if not isinstance(convs, list) or len(convs) < 2:
                continue
        except Exception:
            skipped_parse += 1
            continue

        # Normalizar roles
        roles_map = {"human": "user", "gpt": "assistant", "bot": "assistant"}
        normalized = []
        for turn in convs:
            role = turn.get("role", turn.get("from", "")).lower()
            role = roles_map.get(role, role)
            content = turn.get("content", turn.get("value", "")).strip()
            if role in ("user", "assistant") and content:
                normalized.append({"role": role, "content": content})

        if len(normalized) < 2:
            continue

        # Extrair pares user→assistant
        i = 0
        while i < len(normalized) - 1:
            if normalized[i]["role"] == "user" and normalized[i + 1]["role"] == "assistant":
                u_msg = normalized[i]["content"]
                a_msg = normalized[i + 1]["content"]

                if len(u_msg) < 5 or len(a_msg) < 10:
                    i += 2
                    continue

                if has_unsafe_content(a_msg):
                    skipped_unsafe += 1
                    i += 2
                    continue

                h = text_hash(u_msg + a_msg)
                if h in seen:
                    i += 2
                    continue
                seen.add(h)

                lang = detect_language(u_msg)
                domain = classify_domain_women_health(u_msg + " " + a_msg)
                examples.append(make_example(u_msg, a_msg, domain, path.name, language=lang))
            i += 1

    log.info(f"  women-health [train]: {len(examples)} exemplos | pulados inseguros={skipped_unsafe} | erro parse={skipped_parse}")
    return examples


def classify_domain_women_health(text: str) -> str:
    tl = text.lower()
    if any(w in tl for w in ["violence", "abuse", "abuso", "violência", "agressor"]):
        return "violência_doméstica"
    if any(w in tl for w in ["menstrual", "menstruação", "period", "cycle"]):
        return "menstruação"
    if any(w in tl for w in ["pregnancy", "gravidez", "prenatal", "pré-natal", "gestação"]):
        return "obstetrícia"
    if any(w in tl for w in ["mental health", "saúde mental", "depression", "depressão", "anxiety", "ansiedade"]):
        return "saúde_mental"
    if any(w in tl for w in ["contraceptive", "anticoncepcional", "birth control", "condom", "camisinha"]):
        return "planejamento_familiar"
    if any(w in tl for w in ["cancer", "câncer", "tumor", "mammography", "mamografia", "pap smear", "papanicolau"]):
        return "oncologia_preventiva"
    if any(w in tl for w in ["menopause", "menopausa", "climatério", "hot flash"]):
        return "menopausa_climatério"
    return "ginecologia"


def process_domestic_violence(path: Path) -> list[dict]:
    """
    Usa question como user e response como assistant.
    Filtra por quality scores.
    Inclui instrução de segurança no system prompt quando necessário.
    """
    df = pd.read_excel(path, sheet_name="train")
    examples = []
    seen = set()

    # Filtro de qualidade
    mask = (
        (df["correctness"] >= DV_MIN_CORRECTNESS) &
        (df["helpfulness"] >= DV_MIN_HELPFULNESS) &
        (df["coherence"] >= DV_MIN_COHERENCE)
    )
    df_filtered = df[mask].copy()
    log.info(f"  domestic-violence: {len(df_filtered)}/{len(df)} exemplos após filtro de qualidade")

    for _, row in df_filtered.iterrows():
        q = str(row.get("question", "")).strip()
        r = str(row.get("response", "")).strip()

        if is_empty(q) or is_empty(r):
            continue
        if len(q) < 5 or len(r) < 10:
            continue

        h = text_hash(q + r)
        if h in seen:
            continue
        seen.add(h)

        # Enriquecer resposta com nota de segurança se ausente
        safety_footnote = (
            "\n\n⚠️ Lembre-se: se você estiver em risco imediato, ligue para o número 180 "
            "(Central de Atendimento à Mulher) ou 190 (Polícia). Busque um local seguro "
            "e, se possível, conte com a ajuda de uma pessoa de confiança. "
            "Você não está sozinha e a culpa não é sua."
        )
        if "180" not in r and "delegacia" not in r.lower() and "segurança" not in r.lower():
            r = r + safety_footnote

        lang = detect_language(q)
        examples.append(make_example(q, r, "violência_doméstica", path.name, language=lang))

    log.info(f"  domestic-violence: {len(examples)} exemplos gerados")
    return examples


def process_menstrual_health(path: Path, split: str) -> list[dict]:
    """
    Usa instruction (string) como user e output (string) como assistant.
    """
    df = pd.read_excel(path, sheet_name=split)
    inst_col = "instruction (string)"
    out_col = "output (string)"
    examples = []
    seen = set()

    for _, row in df.iterrows():
        inst = str(row.get(inst_col, "")).strip()
        output = str(row.get(out_col, "")).strip()

        if is_empty(inst) or is_empty(output):
            continue
        if len(inst) < 5 or len(output) < 10:
            continue

        h = text_hash(inst + output)
        if h in seen:
            continue
        seen.add(h)

        lang = detect_language(inst)
        examples.append(make_example(inst, output, "menstruação", path.name, language=lang))

    log.info(f"  menstrual-health [{split}]: {len(examples)} exemplos extraídos")
    return examples


# ---------------------------------------------------------------------------
# Balanceamento de domínios
# ---------------------------------------------------------------------------

def balance_domains(examples: list[dict], max_per_domain: int = 3000) -> list[dict]:
    """
    Evita que um único domínio domine o dataset de treino.
    Limita cada domínio a max_per_domain exemplos,
    mantendo proporção relativa se o domínio tiver menos.
    """
    by_domain = defaultdict(list)
    for ex in examples:
        d = ex["metadata"]["domain"]
        by_domain[d].append(ex)

    log.info("Distribuição antes do balanceamento:")
    for domain, exs in sorted(by_domain.items(), key=lambda x: -len(x[1])):
        log.info(f"  {domain}: {len(exs)}")

    balanced = []
    for domain, exs in by_domain.items():
        random.shuffle(exs)
        balanced.extend(exs[:max_per_domain])

    random.shuffle(balanced)
    log.info(f"Total após balanceamento: {len(balanced)}")
    return balanced


# ---------------------------------------------------------------------------
# Escrita JSONL
# ---------------------------------------------------------------------------

def write_jsonl(examples: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info(f"Escrito: {path} ({len(examples)} exemplos)")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_preprocessing() -> dict:
    log.info("=" * 60)
    log.info("INICIANDO PRÉ-PROCESSAMENTO")
    log.info("=" * 60)

    # ---- Coletar todos os exemplos ----
    all_train: list[dict] = []
    all_val: list[dict] = []
    all_test: list[dict] = []

    # 1. Musicotherapy / Obstetric
    p = EXCEL_FILES["musicotherapy_obstetric"]
    all_train.extend(process_musicotherapy(p, "train"))
    all_val.extend(process_musicotherapy(p, "validation"))

    # 2. Women Health Mini
    p = EXCEL_FILES["women_health_mini"]
    wh_examples = process_women_health(p)
    # Split 90/10 pois não há split explícito
    cut = int(len(wh_examples) * 0.9)
    all_train.extend(wh_examples[:cut])
    all_val.extend(wh_examples[cut:])

    # 3. Domestic Violence
    p = EXCEL_FILES["domestic_violence"]
    dv_examples = process_domestic_violence(p)
    # Todo no train (poucos exemplos)
    all_train.extend(dv_examples)

    # 4. Menstrual Health
    p = EXCEL_FILES["menstrual_health"]
    all_train.extend(process_menstrual_health(p, "train"))
    all_test.extend(process_menstrual_health(p, "test"))

    # ---- Remover duplicatas globais ----
    def dedup(examples: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for ex in examples:
            msgs = ex["messages"]
            key = text_hash(msgs[1]["content"] + msgs[2]["content"])
            if key not in seen:
                seen.add(key)
                out.append(ex)
        return out

    all_train = dedup(all_train)
    all_val = dedup(all_val)
    all_test = dedup(all_test)

    # ---- Balancear domínios no train ----
    all_train = balance_domains(all_train, max_per_domain=3000)

    # ---- Embaralhar ----
    random.shuffle(all_train)
    random.shuffle(all_val)

    # ---- Escrever JSONL ----
    from config import JSONL_TRAIN, JSONL_VALIDATION, JSONL_TEST
    write_jsonl(all_train, JSONL_TRAIN)
    write_jsonl(all_val, JSONL_VALIDATION)
    write_jsonl(all_test, JSONL_TEST)

    # ---- Estatísticas ----
    def domain_dist(examples):
        c = Counter(ex["metadata"]["domain"] for ex in examples)
        return dict(c.most_common())

    def risk_dist(examples):
        c = Counter(ex["metadata"]["risk_level"] for ex in examples)
        return dict(c.most_common())

    def lang_dist(examples):
        c = Counter(ex["metadata"]["language"] for ex in examples)
        return dict(c.most_common())

    unsafe_train = sum(1 for ex in all_train if ex["metadata"]["unsafe_flag"])

    report = {
        "generated_at": datetime.now().isoformat(),
        "splits": {
            "train": {
                "total": len(all_train),
                "domain_distribution": domain_dist(all_train),
                "risk_distribution": risk_dist(all_train),
                "language_distribution": lang_dist(all_train),
                "unsafe_flagged": unsafe_train,
            },
            "validation": {
                "total": len(all_val),
                "domain_distribution": domain_dist(all_val),
                "risk_distribution": risk_dist(all_val),
                "language_distribution": lang_dist(all_val),
            },
            "test": {
                "total": len(all_test),
                "domain_distribution": domain_dist(all_test),
            },
        },
        "output_files": {
            "train": str(JSONL_TRAIN),
            "validation": str(JSONL_VALIDATION),
            "test": str(JSONL_TEST),
        },
        "notes": [
            "Exemplos com unsafe_flag=true foram mantidos no dataset mas marcados para revisão manual.",
            "Todos os exemplos de violência doméstica recebem nota de segurança quando não contêm referência a serviços de apoio.",
            "Dados de menstrual_health (test) foram separados para avaliação clínica.",
        ],
    }

    report_path = REPORTS_DIR / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    log.info(f"PRÉ-PROCESSAMENTO CONCLUÍDO")
    log.info(f"  Train:      {len(all_train)} exemplos")
    log.info(f"  Validation: {len(all_val)} exemplos")
    log.info(f"  Test:       {len(all_test)} exemplos")
    log.info(f"Relatório: {report_path}")
    log.info("=" * 60)

    return report


if __name__ == "__main__":
    report = run_preprocessing()
    print("\n=== RESUMO DO PRÉ-PROCESSAMENTO ===")
    for split, info in report["splits"].items():
        print(f"\n[{split.upper()}] {info['total']} exemplos")
        if "domain_distribution" in info:
            for d, c in info["domain_distribution"].items():
                print(f"  {d}: {c}")
