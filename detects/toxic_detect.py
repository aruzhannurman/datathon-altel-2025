from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os
import re
from typing import List, Dict, Any, Set, Tuple

try:
    import Levenshtein

    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    print("⚠️ Levenshtein library not available, using basic string matching only")

# --- Global variables for model caching ---
_model = None
_tokenizer = None
_bad_words = None
_toxic_id = None
_device = None


def _load_toxic_model():
    """Load the toxic detection model and profane words (cached)."""
    global _model, _tokenizer, _bad_words, _toxic_id, _device

    if _model is None:
        MODEL = "sismetanin/rubert-toxic-pikabu-2ch"
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL)

        id2label = _model.config.id2label
        _toxic_id = next((i for i, n in id2label.items() if "toxic" in n.lower()), 1)

        _device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        _model.to(_device).eval()

        # Load profane words
        profane_file = "profane_words.txt"
        if os.path.exists(profane_file):
            with open(profane_file, "r", encoding="utf-8") as f:
                _bad_words = set(w.strip().lower() for w in f if w.strip())
        else:
            _bad_words = set()

    return _model, _tokenizer, _bad_words, _toxic_id, _device


def _normalize_text(text: str) -> str:
    """Normalize text for better matching by removing special characters and digits."""
    # Replace common character substitutions
    substitutions = {
        "@": "а",
        "4": "а",
        "3": "е",
        "1": "и",
        "0": "о",
        "6": "б",
        "9": "д",
        "$": "s",
        "5": "s",
        "7": "т",
        "8": "в",
        "!": "i",
        "+": "т",
    }

    normalized = text.lower()
    for old, new in substitutions.items():
        normalized = normalized.replace(old, new)

    # Handle asterisk censoring (replace * with common vowels/consonants for matching)
    # This helps match "бл*дь" with "блядь" and "ни*уя" with "нихуя"
    normalized = re.sub(r"бл\*", "блядь", normalized)  # бл* -> блядь
    normalized = re.sub(r"ни\*уя", "нихуя", normalized)  # ни*уя -> нихуя
    normalized = re.sub(r"х\*й", "хуй", normalized)  # х*й -> хуй
    normalized = re.sub(r"п\*здец", "пиздец", normalized)  # п*здец -> пиздец
    normalized = re.sub(r"еб\*", "ебать", normalized)  # еб* -> ебать

    # General asterisk replacement (try common letters)
    normalized = re.sub(r"\*", "а", normalized)  # Replace remaining * with 'а'

    # Remove repeated characters (e.g., "сукаааа" -> "сука")
    normalized = re.sub(r"(.)\1{2,}", r"\1", normalized)

    return normalized


def _find_profane_substrings(
    text: str, profane_words: Set[str]
) -> List[Tuple[str, str]]:
    """Find profane words as substrings in the text."""
    found_profane = []
    normalized_text = _normalize_text(text)

    for profane_word in profane_words:
        if (
            len(profane_word) >= 3
        ):  # Only check words with 3+ characters for substring matching
            if profane_word in normalized_text:
                found_profane.append((profane_word, "substring"))

    return found_profane


def _find_fuzzy_profane_words(
    text: str, profane_words: Set[str], max_distance: int = 1
) -> List[Tuple[str, str, int]]:
    """Find profane words using fuzzy matching with Levenshtein distance."""
    if not LEVENSHTEIN_AVAILABLE:
        return []

    found_profane = []
    normalized_text = _normalize_text(text)
    words = re.findall(r"\b\w+\b", normalized_text)

    for word in words:
        if len(word) >= 3:  # Only check words with 3+ characters
            for profane_word in profane_words:
                if len(profane_word) >= 3:
                    distance = Levenshtein.distance(word, profane_word)
                    # Allow more distance for longer words
                    threshold = (
                        max_distance if len(profane_word) <= 5 else max_distance + 1
                    )

                    if (
                        distance <= threshold and distance > 0
                    ):  # distance > 0 to avoid exact matches
                        found_profane.append((profane_word, word, distance))

    return found_profane


def _detect_advanced_profanity(
    text: str, profane_words: Set[str]
) -> Tuple[bool, float, str]:
    """
    Advanced profanity detection using multiple techniques:
    1. Exact word matching
    2. Substring matching (profane word contained in other words)
    3. Fuzzy matching using Levenshtein distance for distorted words

    Returns:
        has_profane (bool): Whether profane content was detected
        profane_probability (float): Probability score (0.0-1.0) based on match types
        reason (str): Detailed explanation of matches
    """
    if not profane_words:
        return False, 0.0, "No profane words loaded"

    reasons = []
    profane_score = 0.0

    # 1. Exact word matching (highest confidence: 1.0)
    text_words = set(re.findall(r"\b\w+\b", text.lower()))
    exact_matches = text_words.intersection(profane_words)
    if exact_matches:
        reasons.append(f"Exact matches: {', '.join(list(exact_matches)[:3])}")
        profane_score = max(profane_score, 1.0)  # Highest confidence

    # 2. Substring matching (medium-high confidence: 0.8)
    substring_matches = _find_profane_substrings(text, profane_words)
    if substring_matches:
        matches = [match[0] for match in substring_matches[:3]]
        reasons.append(f"Substring matches: {', '.join(matches)}")
        profane_score = max(profane_score, 0.8)  # High confidence

    # 3. Fuzzy matching for distorted words (medium confidence: 0.6-0.7 based on distance)
    fuzzy_matches = _find_fuzzy_profane_words(text, profane_words, max_distance=1)
    if fuzzy_matches:
        matches = [f"{match[1]}~{match[0]}" for match in fuzzy_matches[:3]]
        reasons.append(f"Fuzzy matches: {', '.join(matches)}")
        # Score based on edit distance (closer = higher confidence)
        min_distance = min(match[2] for match in fuzzy_matches)
        fuzzy_confidence = 0.7 if min_distance == 1 else 0.6
        profane_score = max(profane_score, fuzzy_confidence)

    has_profane = profane_score > 0.0
    reason = "; ".join(reasons) if reasons else "No profane words detected"

    return has_profane, profane_score, reason


def detect_toxicity_batch(
    texts: List[str], threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Detect toxicity for a batch of texts.

    Args:
        texts: List of texts to analyze
        threshold: Toxicity threshold (default 0.5)

    Returns:
        List of dicts with 'is_toxic' (bool), 'toxic_score' (float), and 'reason' (str)
    """
    model, tok, bad_words, toxic_id, device = _load_toxic_model()

    results = []

    with torch.no_grad():
        enc = tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

        for text, p in zip(texts, probs):
            # Step 1: Advanced profanity detection with probability
            has_profane, profane_probability, profane_reason = (
                _detect_advanced_profanity(text, bad_words)
            )

            # Step 2: Get model toxicity probability
            model_probability = float(p[toxic_id])

            # Step 3: Optimal combination using confidence-based weighting
            if has_profane:
                # Confidence-based weighting: higher profanity confidence → higher profanity weight
                profane_confidence = (
                    profane_probability  # 0.6-1.0 for profanity matches
                )
                model_confidence = model_probability  # 0.0-1.0 for model prediction

                # Dynamic weight calculation based on profanity confidence
                if (
                    profane_confidence >= 0.8
                ):  # High profanity confidence (exact/substring matches)
                    profane_weight = 0.8  # 80% profanity, 20% model
                elif (
                    profane_confidence >= 0.6
                ):  # Medium profanity confidence (fuzzy matches)
                    profane_weight = 0.7  # 70% profanity, 30% model
                else:
                    profane_weight = 0.6  # 60% profanity, 40% model

                model_weight = 1.0 - profane_weight

                # Weighted combination
                combined_probability = (profane_weight * profane_probability) + (
                    model_weight * model_probability
                )

                # Enhanced reasoning
                reason_parts = [
                    f"Profane: {profane_reason} (conf={profane_confidence:.2f})",
                    f"Model: {model_probability:.3f} (conf={model_confidence:.2f})",
                ]
                combined_reason = f"Weighted ({profane_weight:.1f}/{model_weight:.1f}, final={combined_probability:.3f}): {'; '.join(reason_parts)}"
            else:
                # No profanity detected - use model prediction with slight boost for edge cases
                combined_probability = model_probability
                combined_reason = f"Model prediction: {model_probability:.3f}"

            # Final decision based on standard threshold (most conservative)
            # Use same threshold for all cases - no special treatment for profanity detection
            adaptive_threshold = (
                threshold  # Same threshold for all (most conservative approach)
            )
            is_toxic = combined_probability >= adaptive_threshold

            results.append(
                {
                    "is_toxic": is_toxic,
                    "toxic_score": combined_probability,
                    "profane_probability": profane_probability,
                    "model_probability": model_probability,
                    "adaptive_threshold": (
                        adaptive_threshold if has_profane else threshold
                    ),
                    "reason": combined_reason,
                }
            )

    return results


def detect_toxicity_single(text: str, threshold: float = 0.6) -> Dict[str, Any]:
    """
    Detect toxicity for a single text.

    Args:
        text: Text to analyze
        threshold: Toxicity threshold (default 0.5)

    Returns:
        Dict with 'is_toxic' (bool), 'toxic_score' (float), and 'reason' (str)
    """
    return detect_toxicity_batch([text], threshold)[0]


if __name__ == "__main__":
    # --- Input texts to check (enhanced test cases) ---
    texts = [
        # Neutral / benign
        "это оскорбительно",
        "меня оскорбили в магазине",
        "Интернет просто жуткий, не работает вообще",
        "Интернет отстой, не могу ничего скачать",
        "со мной грубо обращались консультанты",
        # Original profane content
        "Ля Я того рот, самый хреновый интернет, до операторов хер дозвониться",
        "Сукаааааа коментаридың өзін 2 минутта әрең ашад сеть қашан дұрысталад",
        "СВЯЗЬ ТУПОЙ ИНТЕРНЕТ ТУПОЙ ,А ТАРИФ КОСМИЧЕСКИЙ",
        "У вас такой интернет конченый",
        # Test cases for enhanced detection
        # 1. Repeated characters (should be caught by normalization)
        "блядддь какой плохой сервис",
        "сукаааааа не работает",
        # 2. Character substitutions (should be caught by normalization)
        "бл@дь этот интернет",
        "сук4 какая связь",
        "х3рня полная",
        # 3. Substring matching (profane words inside other words)
        "засранец этот оператор",
        "дебилизм какой-то",
        # 4. Fuzzy matching (typos and distortions)
        "блять что за сервис",  # typo in profane word
        "сука что происходит",  # exact match for comparison
        # 5. Censored profanity (asterisks)
        "Что с интернетом бл*? Последний 2-3 дня не грузит ни*уя. За тариф берете бешенные деньги. А работать ни*уя не хотите. До оператора не дозвониться",
    ]

    # Use the new batch function
    results = detect_toxicity_batch(texts, threshold=0.5)

    print("🔍 Enhanced Toxicity Detection Results:")
    print("=" * 80)

    # Convert to the original format for compatibility and show detailed results
    formatted_results = []
    for i, (text, result) in enumerate(zip(texts, results)):
        binary = 0 if result["is_toxic"] else 1
        formatted_results.append(
            {
                "text": text[:50] + "..." if len(text) > 50 else text,
                "binary": binary,
                "toxic_score": f"{result['toxic_score']:.3f}",
                "reason": result["reason"],
            }
        )

        # Detailed output for demonstration
        status = "🔴 TOXIC" if result["is_toxic"] else "🟢 CLEAN"
        threshold_used = result.get("adaptive_threshold", 0.5)
        print(
            f"{i+1:2d}. {status} | Combined: {result['toxic_score']:.3f} | Threshold: {threshold_used:.3f} | Profane: {result['profane_probability']:.3f} | Model: {result['model_probability']:.3f}"
        )
        print(f"    Text: {text}")
        print(f"    Reason: {result['reason']}")
        print()

    # --- For nice table view ---
    df = pd.DataFrame(formatted_results)
    print("\n📊 Summary Table:")
    print(df.to_string(index=False))
