import os, sys, json, argparse
from openai import OpenAI

TARGET_LABELS = ["question", "review", "complaint", "gratitude"]
TONE_LABELS = ["negative", "neutral", "positive"]

from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModel

RUBERT_MODEL_ID = "cointegrated/rubert-tiny2"
LABELS_EN = ["question", "review", "complaint", "gratitude"]
TONES_EN = ["negative", "neutral", "positive"]

LABEL_DESCRIPTIONS_RU = {
    "question": "вопрос, просьба пояснить, как сделать",
    "review": "отзыв, оценка качества, впечатление о связи",
    "complaint": "жалоба, претензия, недовольство и проблемы",
    "gratitude": "благодарность, спасибо, положительная оценка",
}

TONE_DESCRIPTIONS_RU = {
    "negative": "негативный эмоциональный тон, жалобы, претензии, недовольство",
    "neutral": "нейтральный тон, без ярко выраженных эмоций",
    "positive": "позитивный тон, похвала, благодарность, одобрение",
}

SYSTEM_PROMPT_TEMPLATE = """
You are a careful classifier of user comments.

Your task is to assign the comment to exactly one of the categories:
- "question"
- "review"
- "complaint"
- "gratitude"

Also assign the tone:
- "negative"
- "neutral"
- "positive"

Output ONLY valid JSON with schema:
{"label": one_of(["question","review","complaint","gratitude"]),
 "tone": one_of(["negative","neutral","positive"]),
 "confidence": 0..1,
 "reason": "short explanation"}

Here are some examples:

Example:
TEXT: "Почему интернет не работает?"
LABEL: question
TONE: neutral

Example:
TEXT: "Здравствуйте! Неделю назад заявку оставлял! Исходящие звонки не осуществляются, не могу позвонить даже в Каспий банк? Что за отношение у вас клиентам, в колл центр не дозвонишься, сбрасывают! Все сложно у вас, нет никакой оперативности, обслуживание на нуле 🥺"
LABEL: complaint
TONE: negative

Example:
TEXT: "Что за интернет? Его вообще нет. Просто ужасная связь!!!"
LABEL: complaint
TONE: negative

Example:
TEXT: "Ни в коем случае не покупайте эту связь, самая худшая связь👎👎👎"
LABEL: complaint
TONE: negative

Example:
TEXT: "Пользуюсь Altel уже несколько лет — всегда стабильная связь и быстрый интернет! 👍"
LABEL: review
TONE: positive

Example:
TEXT: "Очень доволен оператором Altel: тарифы доступные, а качество связи на высоте 💯"
LABEL: review
TONE: positive

Example:
TEXT: "Altel приятно удивил — обслуживание быстрое, сотрудники всегда помогают, интернет работает отлично 😊"
LABEL: review
TONE: positive

Example:
TEXT: "Спасибо Altel за качественную работу, приятно пользоваться вашими услугами!"
LABEL: gratitude
TONE: positive

Example:
TEXT: "Как подключить новый тариф Altel и можно ли это сделать онлайн?"
LABEL: question
TONE: neutral

Example:
TEXT: "Благодарю за быструю помощь в колл-центре, решили мой вопрос за пару минут 🙏"
LABEL: gratitude
TONE: positive

Example:
TEXT: "Подскажите, пожалуйста, как проверить остаток интернета по моему тарифу?"
LABEL: question
TONE: neutral

Example:
TEXT: "Altel отличный оператор, связь ловит даже там, где другие не работают!"
LABEL: review
TONE: positive

Example:
TEXT: "Алтел топчик, интернет летает 🔥"
LABEL: review
TONE: positive

Example:
TEXT: "Эй, у кого ещё сеть тупит? Уже час вообще не ловит 😡"
LABEL: complaint
TONE: negative

Example:
TEXT: "Спасибо за нормальную скорость, теперь видосы грузятся быстро 🙌"
LABEL: gratitude
TONE: positive

Example:
TEXT: "Алтел, ну как так? Почему у меня опять минус баланс? 🤔"
LABEL: question
TONE: negative

Example:
TEXT: "Кстати, связь у вас лучше, чем у многих конкурентов, респект 👍"
LABEL: review
TONE: positive

Example:
TEXT: "О, класс, опять интернета нет… спасибо, Altel, всегда вовремя 👏"
LABEL: complaint
TONE: negative

Example:
TEXT: "Лучший сервис! Чтобы дозвониться в поддержку, нужно иметь целый день свободного времени 😂"
LABEL: complaint
TONE: negative

Example:
TEXT: "Altel, вы гении! Баланс исчезает быстрее, чем интернет грузит страницы 🤡"
LABEL: complaint
TONE: negative

Example:
TEXT: "Ого, интернет включился! Запишите этот день в календарь 🎉"
LABEL: review
TONE: negative

Example:
TEXT: "Поддержка такая быстрая, что я уже состарился, пока ждал ответа 🙄"
LABEL: complaint
TONE: negative

Example:
TEXT: "Создали ситуацию что до Вас никаким каналом недостучаться и еще отчитываетесь как все круто автоматизировали и за это бонусы получаете"
LABEL: complaint
TONE: negative

Example:
TEXT: "@altel_kz вы издеваетесь, да?! До вас не дозвониться, роутер не работает на 5 джи, только 4 джи, элементарно видео по ватсап не могу отправить !!! Беспредел полнейший, в колл центр к вам не дозвониться!!!"
LABEL: complaint
TONE: negative

Example:
TEXT: "Самый отвратительный сервис обслуживания клиентов. Дозвонится невозможно, бесполезный бот в Телеграм, непрозрачные обманные акции. Разочарован. После 15 лет буду менять оператор. @altel_kz - разочарование"
LABEL: complaint
TONE: negative

Example:
TEXT: "Алтел это не связь а беда какая то"
LABEL: review
TONE: negative

Example:
TEXT: "Қалай тарифім бітеді номерді мусорға лақтырам, орны сол жерде, билайнға көшем"
LABEL: review
TONE: negative

Example:
TEXT: "Қалай тарифім бітеді номерді мусорға лақтырам, орны сол жерде, билайнға көшем"
LABEL: review
TONE: negative

Example:
TEXT: "Почему у вас интернет тупая"
LABEL: question
TONE: negative

Example:
TEXT: "Интернет қашан дұрыс жасайды мүлде сеть жоқ"
LABEL: complaint
TONE: negative

Example:
TEXT: "Отвратительное отношение к аборигенам!
Ничего не работает, а оплата за тариф снимается мгновенно.
Дозвониться не возможно.
Каждый день вы теряете своих абонентов с таким отношением.
Благо есть другие операторы сотовой связи. Удачи"
LABEL: complaint
TONE: negative


---

Now classify this new comment:
TEXT: "{text}"
"""


def build_prompt(text: str) -> str:
    # We only need to format the template with the current input text
    return SYSTEM_PROMPT_TEMPLATE.replace("{text}", text)


def classify_comment(
    text: str,
    model: str = "gpt-4o-mini-2024-07-18",
    temperature: float = 0.0,
    max_retries: int = 1,
) -> dict:
    client = OpenAI()
    messages = [{"role": "system", "content": build_prompt(text)}]

    last_err = None
    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, messages=messages
            )
            raw = resp.choices[0].message.content.strip()
            try:
                obj = json.loads(raw)
                if obj.get("label") in TARGET_LABELS and obj.get("tone") in TONE_LABELS:
                    return obj
                last_err = f"Validation failed: {raw}"
            except json.JSONDecodeError as e:
                last_err = f"JSON parse error: {e}"
        except Exception as e:
            last_err = f"OpenAI API error: {e}"
    raise ValueError(f"Failed to obtain valid JSON. Last error: {last_err}")


_rubert = None
_rubert_tok = None


def _load_rubert() -> Tuple[AutoTokenizer, AutoModel]:
    global _rubert, _rubert_tok
    if _rubert is None:
        _rubert_tok = AutoTokenizer.from_pretrained(RUBERT_MODEL_ID)
        _rubert = AutoModel.from_pretrained(RUBERT_MODEL_ID)
        _rubert.eval()
    return _rubert_tok, _rubert


@torch.no_grad()
def _mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def _embed(texts: List[str]) -> torch.Tensor:
    tok, mdl = _load_rubert()
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    out = mdl(**enc)
    emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # cosine
    return emb


def _softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max()
    return torch.softmax(x, dim=-1)


def classify_with_rubert(text: str) -> Dict:
    """Zero-shot-ish: cosine similarity between text and RU descriptions for labels & tones."""
    label_candidates = [LABEL_DESCRIPTIONS_RU[k] for k in LABELS_EN]
    tone_candidates = [TONE_DESCRIPTIONS_RU[k] for k in TONES_EN]
    embs = _embed([text] + label_candidates + tone_candidates)

    q = embs[0:1]
    label_mat = embs[1 : 1 + len(LABELS_EN)]
    tone_mat = embs[1 + len(LABELS_EN) :]

    sim_labels = (q @ label_mat.T).squeeze(0)  # (4,)
    sim_tones = (q @ tone_mat.T).squeeze(0)  # (3,)

    p_labels = _softmax(sim_labels)
    p_tones = _softmax(sim_tones)

    li = int(torch.argmax(p_labels))
    ti = int(torch.argmax(p_tones))

    label = LABELS_EN[li]
    tone = TONES_EN[ti]
    # Combine heads conservatively (geometric mean)
    conf = float(torch.sqrt(p_labels[li] * p_tones[ti]).item())

    return {
        "label": label,
        "tone": tone,
        "confidence": round(conf, 3),
        "reason": (
            f"RuBERT-tiny2 cosine similarity: "
            f"label={label} (p≈{float(p_labels[li]):.2f}), "
            f"tone={tone} (p≈{float(p_tones[ti]):.2f})"
        ),
    }


def classify_comment_with_verification(
    text: str,
    model: str = "gpt-4o-mini-2024-07-18",
    strict: bool = False,
    margin: float = 0.20,
) -> Dict:
    llm_res = classify_comment(text, model=model)  # your existing LLM function
    rubert_res = classify_with_rubert(text)
    return reconcile_predictions(llm_res, rubert_res, strict=strict, margin=margin)


def reconcile_predictions(
    llm_res: Dict, rubert_res: Dict, strict: bool = False, margin: float = 0.20
) -> Dict:
    """
    Combine LLM (primary) with RuBERT (verification).

    - If both agree -> final = that label/tone; confidence = max(LLM, RuBERT).
    - If disagree:
        * default (strict=False): keep LLM, lower confidence slightly, note discrepancy.
        * strict=True: RuBERT may override if (rubert_conf - llm_conf) >= margin.
    """
    out = {
        "label_llm": llm_res.get("label"),
        "tone_llm": llm_res.get("tone"),
        "confidence_llm": float(llm_res.get("confidence", 0.0)),
        "label_rubert": rubert_res.get("label"),
        "tone_rubert": rubert_res.get("tone"),
        "confidence_rubert": float(rubert_res.get("confidence", 0.0)),
    }

    agree_label = out["label_llm"] == out["label_rubert"]
    agree_tone = out["tone_llm"] == out["tone_rubert"]
    out["agree_label"] = agree_label
    out["agree_tone"] = agree_tone

    # Default: keep LLM
    final_label = out["label_llm"]
    final_tone = out["tone_llm"]
    final_conf = out["confidence_llm"]
    reason = "LLM primary."

    if agree_label and agree_tone:
        final_conf = max(out["confidence_llm"], out["confidence_rubert"])
        reason = "LLM & RuBERT agree."
    else:
        if strict and (out["confidence_rubert"] - out["confidence_llm"] >= margin):
            final_label = out["label_rubert"]
            final_tone = out["tone_rubert"]
            final_conf = out["confidence_rubert"]
            reason = f"RuBERT override (margin ≥ {margin})."
        else:
            # keep LLM but discount a bit for disagreement
            final_conf = max(0.0, final_conf - 0.1)
            reason = "Disagreement: kept LLM; discounted confidence."

    out["final_label"] = final_label
    out["final_tone"] = final_tone
    out["final_confidence"] = round(final_conf, 3)
    out["reason"] = reason
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Classify a comment: label (question/review/complaint/gratitude) + tone (negative/neutral/positive)"
    )
    ap.add_argument("--text", type=str, help="Single comment text")
    ap.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    ap.add_argument(
        "--verify", action="store_true", help="Run RuBERT-tiny2 verification after LLM"
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Allow RuBERT to override LLM if much more confident",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=0.20,
        help="Confidence margin for strict override",
    )
    args = ap.parse_args()

    if args.text:
        if args.verify:
            res = classify_comment_with_verification(
                args.text, model=args.model, strict=args.strict, margin=args.margin
            )
        else:
            res = classify_comment(args.text, model=args.model)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            if args.verify:
                res = classify_comment_with_verification(
                    text, model=args.model, strict=args.strict, margin=args.margin
                )
            else:
                res = classify_comment(text, model=args.model)
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return

    ap.print_help()


if __name__ == "__main__":
    main()
