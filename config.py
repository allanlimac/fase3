import os
from pathlib import Path


BASE_DIR = Path(__file__).parent.resolve()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "model_output"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR, MODEL_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


EXCEL_FILES = {
    "musicotherapy_obstetric": BASE_DIR / "musicotherapy_obstetric_dataset.xlsx",
    "women_health_mini": BASE_DIR / "women-health-mini.xlsx",
    "domestic_violence": BASE_DIR / "domestic-violence-dataset-prep.xlsx",
    "menstrual_health": BASE_DIR / "menstrual_health_awareness_dataset.xlsx",
}


JSONL_TRAIN = PROCESSED_DIR / "train.jsonl"
JSONL_VALIDATION = PROCESSED_DIR / "validation.jsonl"
JSONL_TEST = PROCESSED_DIR / "test.jsonl"
JSONL_CLINICAL_TESTS = PROCESSED_DIR / "clinical_tests.jsonl"


SYSTEM_PROMPT = (
    "Você é uma assistente virtual especializada em saúde da mulher, com foco em "
    "ginecologia, obstetrícia, saúde reprodutiva, violência doméstica, saúde mental, "
    "ciclo menstrual, menopausa, pré-natal, puerpério e amamentação.\n\n"
    "Responda sempre em português brasileiro, com linguagem clara, acolhedora, segura "
    "e baseada em evidências. Você pode oferecer educação em saúde, explicar sintomas, "
    "exames e procedimentos, orientar sobre sinais de alerta e sugerir quando procurar "
    "atendimento médico.\n\n"
    "Você NÃO deve:\n"
    "- Dar diagnóstico definitivo.\n"
    "- Prescrever medicamentos, doses ou terapias individualizadas.\n"
    "- Substituir consulta médica.\n"
    "- Minimizar sintomas graves.\n"
    "- Julgar pacientes.\n"
    "- Dar orientações que aumentem o risco de uma vítima de violência doméstica.\n"
    "- Inventar protocolos, exames ou condutas.\n\n"
    "Sempre recomende atendimento imediato se houver sinais de emergência: sangramento "
    "intenso, dor forte, febre persistente, desmaio, convulsões, falta de ar, dor no "
    "peito, ideação suicida, sangramento na gravidez, sinais de pré-eclâmpsia, "
    "sangramento pós-parto intenso ou risco iminente de violência.\n\n"
    "Ao responder, siga esta estrutura:\n"
    "1. Acolha a pessoa.\n"
    "2. Explique de forma simples.\n"
    "3. Liste sinais de alerta, se houver.\n"
    "4. Oriente próximos passos seguros.\n"
    "5. Reforce que a resposta não substitui avaliação profissional."
)


DOMAINS = [
    "ginecologia",
    "obstetrícia",
    "violência_doméstica",
    "menstruação",
    "saúde_mental",
    "planejamento_familiar",
    "oncologia_preventiva",
    "menopausa_climatério",
    "neonatologia",
    "amamentação",
]

RISK_LEVELS = ["baixo", "moderado", "alto", "emergência"]

EMERGENCY_KEYWORDS = [
    "sangramento intenso", "sangramento na gravidez", "hemorragia",
    "convulsão", "convulsões", "desmaio", "perda de consciência",
    "falta de ar", "dificuldade para respirar", "dor no peito",
    "pressão alta", "pré-eclâmpsia", "eclâmpsia",
    "ideação suicida", "suicídio", "quero me matar", "me machucar",
    "movimentos fetais", "bebê não mexe", "mexendo menos", "mexe menos",
    "violência imediata", "risco de vida", "sangramento pós-parto",
    "febre alta", "sepse", "infecção grave",
    "encharcando absorvente", "absorvente por hora", "sangramento abundante",
    "bebê parando de mexer", "redução dos movimentos",
]

HIGH_RISK_KEYWORDS = [
    "sangramento", "dor abdominal intensa", "dor pélvica forte",
    "febre", "corrimento com odor", "nódulo mamário",
    "resultado alterado", "papanicolau alterado", "violência",
    "agressão", "depressão pós-parto", "tristeza extrema",
    "ansiedade grave", "vômitos frequentes",
]

MODERATE_RISK_KEYWORDS = [
    "dor pélvica", "corrimento", "irregularidade menstrual",
    "cólica intensa", "sangramento entre períodos", "enjoo",
    "preocupação", "anticoncepcional", "sintoma", "efeito colateral",
]


DV_MIN_CORRECTNESS = 3.0
DV_MIN_HELPFULNESS = 3.0
DV_MIN_COHERENCE = 3.0


FINETUNE_CONFIG = {
    # Modelo base — Gemma 2 2B escolhido por:
    # - Não requer aprovação (diferente do LLaMA da Meta)
    # - Leve o suficiente para 4GB VRAM com QLoRA
    # - Boa qualidade para português
    "base_model": "google/gemma-2-2b-it",
    # Alternativas (requerem aprovação ou mais VRAM):
    # "meta-llama/Llama-3.2-3B-Instruct"   (requer aceite nos termos da Meta no HF)
    # "meta-llama/Llama-3.1-8B-Instruct"   (8GB+ VRAM com QLoRA)
    # "mistralai/Mistral-7B-Instruct-v0.3" (6GB+ VRAM com QLoRA)
    # "Qwen/Qwen2.5-7B-Instruct"           (6GB+ VRAM com QLoRA)

    # LoRA / QLoRA — ajustado para RTX 3050 Laptop (4GB VRAM)
    "use_qlora": True,            # True = QLoRA (4-bit), False = LoRA (16-bit)
    "lora_r": 8,                  # Rank reduzido para caber em 4GB VRAM (original: 16)
    "lora_alpha": 16,             # Alpha = 2 * lora_r (boa prática)
    "lora_dropout": 0.05,
    "target_modules": [           # Módulos alvo para LoRA (apenas atenção, sem FFN — economiza VRAM)
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],

    # Treinamento — ajustado para 4GB VRAM
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,   # Reduzido de 2 para 1 (4GB VRAM)
    "per_device_eval_batch_size": 1,    # Reduzido de 2 para 1
    "gradient_accumulation_steps": 4,   # Reduzido de 16 para 4 — menos overhead por step (efetivo: 1*4=4)
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_seq_length": 256,        # Reduzido de 512 para 256 — libera VRAM e acelera treino
    "fp16": False,                # desativado: bf16 é nativo no Gemma 2 e compatível com RTX 3050 Ampere
    "bf16": True,                 # ativado: sem GradScaler, compatível com camadas BFloat16 internas do Gemma 2

    # Avaliação e salvamento
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_steps": 200,
    "logging_steps": 25,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "save_total_limit": 3,

    # Saída
    "output_dir": str(MODEL_DIR / "checkpoints"),
    "final_model_dir": str(MODEL_DIR / "womenhealth-llm-final"),

    # Reprodutibilidade
    "seed": 42,

    # Hugging Face Hub (opcional)
    "push_to_hub": False,
    "hub_model_id": "org/womenhealth-llm",
}


EVAL_CONFIG = {
    "min_emergency_recognition_rate": 0.95,   # 95% mínimo exigido
    "clinical_test_file": str(JSONL_CLINICAL_TESTS),
    "eval_output_report": str(REPORTS_DIR / "clinical_evaluation_report.json"),
}
