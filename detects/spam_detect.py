# pip install openai transformers torch --upgrade
# pip install python-dotenv
import os
import json
from typing import Any, Dict, Tuple

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

# ----------------------------
# OpenAI client
# ----------------------------
try:
    client = OpenAI()  # expects OPENAI_API_KEY in env
    OPENAI_AVAILABLE = True
except Exception as e:
    print(f"Warning: OpenAI client initialization failed: {e}")
    client = None
    OPENAI_AVAILABLE = False

# ----------------------------
# HF secondary spam model
# ----------------------------
_HF_MODEL_ID = "RUSpam/spam_deberta_v4"
_tokenizer = AutoTokenizer.from_pretrained(_HF_MODEL_ID)
_model = AutoModelForSequenceClassification.from_pretrained(_HF_MODEL_ID)
_hf_spam = pipeline("text-classification", model=_model, tokenizer=_tokenizer)

# ----------------------------
# Keyword boost list (RU)
# ----------------------------
SPAM_WORDS_RU = [
    "заработай", "вложи", "инвестиции", "крипто", "лотерея", "скидка",
    "подписывайся", "бесплатно", "акция", "деньги", "прибыль", "доход", "ссылку",
    "ссылке", "переходи", "переходи по ссылке", "кредит", "быстрые деньги",
    "увеличь доход", "пиши в личку", "пиши в директ", "работа на дому",
    "работа для всех", "заработок", "заработок в интернете", "выиграй", "выигрыш",
    "розыгрыш", "казино", "казино онлайн", "играй и выигрывай", "игра на деньги",
    "ставки", "ставки на спорт", "форекс", "бинарные опционы", "форекс брокер",
    "трейдинг", "трейдер", "биржа", "биткоин", "эфириум", "криптовалюта", "nft",
    "обмен валют", "обменник", "дешёвые кредиты", "займ без отказа", "займ онлайн",
    "быстрый займ", "кредит без справок", "кредит онлайн", "кредит без отказа",
    "работа в интернете", "подработка", "подработка на дому", "работа без вложений",
    "удалённая работа", "работа для студентов", "работа для мам в декрете",
    "вакансия", "вакансии", "требуются сотрудники", "требуются менеджеры",
    "даром", "халява", "халяву", "халявные"
]

def apply_keyword_boost_ru(text_ru: str, score: float, boost: float = 0.3) -> float:
    """
    If RU spam keywords appear in the RU text, increase spam probability.
    """
    low = text_ru.lower()
    if any(w in low for w in SPAM_WORDS_RU):
        score = min(1.0, score + boost)
    return score

# ----------------------------
# JSON schema & validator
# ----------------------------
SCHEMA = {
    "type": "object",
    "required": [
        "language", "translated_russian",
        "is_spam_llm", "llm_confidence",
        "is_spam_hf", "hf_confidence",
        "is_spam_final", "relevance_altel_tele2"
    ],
    "properties": {
        "language": {"type": "string"},
        "translated_russian": {"type": "string"},
        "is_spam_llm": {"type": "boolean"},
        "llm_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "is_spam_hf": {"type": "boolean"},
        "hf_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "is_spam_final": {"type": "boolean"},
        "relevance_altel_tele2": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
            "description": "LLM-scored relevance to Altel/Tele2 context (0..1)."
        },
        "spam_reason_llm": {"type": "string"},
        "spam_reason_hf": {"type": "string"},
        "notes": {"type": "string"},
    },
    "additionalProperties": False
}

def validate_result(payload: Dict[str, Any]) -> bool:
    try:
        # required
        for k in SCHEMA["required"]:
            if k not in payload:
                return False
        # types
        if not isinstance(payload["language"], str):
            return False
        if not isinstance(payload["translated_russian"], str):
            return False
        if not isinstance(payload["is_spam_llm"], bool):
            return False
        if not isinstance(payload["is_spam_hf"], bool):
            return False
        for key in ["llm_confidence", "hf_confidence", "relevance_altel_tele2"]:
            v = payload[key]
            if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                return False
        if not isinstance(payload["is_spam_final"], bool):
            return False
        # optionals
        for opt in ["spam_reason_llm", "spam_reason_hf", "notes"]:
            if opt in payload and not isinstance(payload[opt], str):
                return False
        # no extra fields
        extra = set(payload.keys()) - set(SCHEMA["properties"].keys())
        if extra:
            return False
        return True
    except Exception:
        return False

# ----------------------------
# LLM prompts
# ----------------------------
SYSTEM_INSTRUCTIONS = """\
You are a careful text classifier and translator.

TASKS:
1) Detect the original language of the user text. Return ISO 639-1 code only (e.g., 'ru', 'en', 'kk').
2) Translate the text to Russian EXACTLY AS WRITTEN (literal), preserving meaning, named entities, numbers, emojis, URLs, hashtags, and slang.
   Do not summarize, normalize, paraphrase, or add/remove content.
3) Decide if the message is spam (binary). Consider classic spam signals: mass marketing, scams/phishing, crypto/loan/get-rich schemes,
   repeated promotions, link bait, deceptive offers, irrelevant ads, etc.
4) Score the relevance of the message to the context of Altel/Tele2 (Kazakh/KZ telecom operators) on a continuous scale 0..1,
   where 0 means "completely unrelated to Altel/Tele2" and 1 means "directly about Altel/Tele2 services, tariffs, coverage, shops, SIMs, eSIM, MNP, etc."

OUTPUT:
Return ONLY valid JSON (no markdown fences, no commentary) with keys:
{
  "language": "ISO 639-1 string",
  "translated_russian": "string",
  "is_spam_llm": true/false,
  "llm_confidence": 0.0-1.0,
  "relevance_altel_tele2": 0.0-1.0,
  "spam_reason_llm": "short reason"
}
"""

def build_user_prompt(text: str) -> str:
    return f"""\
INPUT TEXT:
{text}

REQUIREMENTS:
- Detect language (ISO 639-1).
- Translate to Russian literally (no paraphrase).
- Classify spam (true/false) with confidence 0..1.
- Score relevance to Altel/Tele2 in [0..1].
- Return ONLY JSON with keys: language, translated_russian, is_spam_llm, llm_confidence, relevance_altel_tele2, spam_reason_llm.
"""

def llm_analyze(text: str,
                model: str = "gpt-4o-mini-2024-07-18",
                temperature: float = 0.1,
                max_retries: int = 1) -> Dict[str, Any]:
    """
    Single-shot LLM call with minimal retry to ensure valid JSON.
    """
    last_err = None
    for attempt in range(max_retries + 1):
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": build_user_prompt(text)}
            ]
        )
        raw = resp.choices[0].message.content.strip()
        try:
            data = json.loads(raw)
            # sanity checks for required keys (subset schema)
            for k in ["language", "translated_russian", "is_spam_llm", "llm_confidence", "relevance_altel_tele2", "spam_reason_llm"]:
                if k not in data:
                    raise ValueError(f"Missing key {k}")
            # clamp/conf
            data["llm_confidence"] = float(max(0.0, min(1.0, float(data["llm_confidence"]))))
            data["relevance_altel_tele2"] = float(max(0.0, min(1.0, float(data["relevance_altel_tele2"]))))
            data["is_spam_llm"] = bool(data["is_spam_llm"])
            return data
        except Exception as e:
            last_err = f"LLM JSON issue: {e}"

    raise ValueError(f"Failed to obtain valid LLM JSON. Last error: {last_err}")

# ----------------------------
# HF spam verification (RU)
# ----------------------------
def hf_spam_check_ru(text_ru: str) -> Tuple[bool, float, str]:
    """
    Classify RU text with RUSpam model; return (is_spam, confidence, reason).
    Applies keyword boost.
    """
    out = _hf_spam(text_ru, truncation=True)[0]
    raw_label = out["label"].lower()
    # Many HF models use labels like 'LABEL_1'/'LABEL_0' or 'spam'/'ham'
    is_spam = ("spam" in raw_label) or ("1" in raw_label)
    conf = float(out["score"])
    # keyword boost
    boosted_conf = apply_keyword_boost_ru(text_ru, conf, boost=0.3)
    if boosted_conf > conf:
        is_spam = True
        reason = f"HF model label={out['label']}, score={conf:.2f}; boosted by RU keywords → {boosted_conf:.2f}"
    else:
        reason = f"HF model label={out['label']}, score={conf:.2f}"
    return is_spam, float(min(1.0, boosted_conf)), reason

# ----------------------------
# Orchestrator
# ----------------------------
def analyze_text(
    text: str,
    model: str = "gpt-4o-mini-2024-07-18"
) -> Dict[str, Any]:
    """
    1) LLM: detect language, literal RU translation, LLM spam + confidence, relevance score.
    2) HF: spam on RU translation + keyword boost.
    3) Final spam = spam if either (LLM or HF) says spam.
    4) Return validated JSON.
    """
    llm = llm_analyze(text, model=model)

    # HF spam check runs on RU translation
    hf_is_spam, hf_conf, hf_reason = hf_spam_check_ru(llm["translated_russian"])

    is_spam_final = bool(llm["is_spam_llm"] or hf_is_spam)

    result = {
        "language": llm["language"],
        "translated_russian": llm["translated_russian"],
        "is_spam_llm": bool(llm["is_spam_llm"]),
        "llm_confidence": float(llm["llm_confidence"]),
        "is_spam_hf": bool(hf_is_spam),
        "hf_confidence": float(hf_conf),
        "is_spam_final": is_spam_final,
        "relevance_altel_tele2": float(llm["relevance_altel_tele2"]),
        "spam_reason_llm": str(llm.get("spam_reason_llm", "")),
        "spam_reason_hf": hf_reason,
        "notes": "Final spam is True if either LLM or HF predicts spam."
    }

    if not validate_result(result):
        # In the unlikely case of schema violation, force a strict minimal fallback
        result = {
            "language": str(llm.get("language", "und")),
            "translated_russian": str(llm.get("translated_russian", "")),
            "is_spam_llm": bool(llm.get("is_spam_llm", False)),
            "llm_confidence": float(max(0.0, min(1.0, float(llm.get("llm_confidence", 0.0))))),
            "is_spam_hf": bool(hf_is_spam),
            "hf_confidence": float(max(0.0, min(1.0, hf_conf))),
            "is_spam_final": bool(llm.get("is_spam_llm", False) or hf_is_spam),
            "relevance_altel_tele2": float(max(0.0, min(1.0, float(llm.get("relevance_altel_tele2", 0.0))))),
            "spam_reason_llm": str(llm.get("spam_reason_llm", "")),
            "spam_reason_hf": hf_reason,
            "notes": "Schema-normalized fallback."
        }
        if not validate_result(result):
            raise ValueError("Output JSON failed validation even after fallback normalization.")

    return result

# ----------------------------
# Demo
# ----------------------------
if __name__ == "__main__":
    samples = [
        "Алтел это не связь, а беда какая-то 😤",
        "🔥 WIN BIG NOW! Click this crypto airdrop link: http://scam.example",
        "Қалай тарифім бітеді? Нөмірді мусорға лақтырам.",
        "Сегодня акция! Увеличь доход без вложений — пиши в директ!"
    ]
    for t in samples:
        out = analyze_text(t)

        llm_label = "СПАМ" if out["is_spam_llm"] else "НЕ СПАМ"
        hf_label = "СПАМ" if out["is_spam_hf"] else "НЕ СПАМ"
        relevancy = out["relevance_altel_tele2"]

        print(f"\nТекст: {t}")
        print(f"  LLM: {llm_label}")
        print(f"  HF : {hf_label}")
        print(f"  Relevance to Altel/Tele2: {relevancy:.2f}")
