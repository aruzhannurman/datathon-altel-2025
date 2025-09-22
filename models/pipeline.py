#!/usr/bin/env python3
"""
Comprehensive Comment Processing Pipeline
Orchestrates toxic detection, spam detection, comment classification, and answer generation.
"""

import asyncio
import pandas as pd
import json
import os
import sys
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings("ignore")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "Warning: python-dotenv not installed. Install with: pip install python-dotenv"
    )

# Import our async API processor
from api.api_requests import AsyncOpenaiProcessor, OpenAIRequest

# Import individual modules
try:
    from detects.toxic_detect import detect_toxicity_batch

    TOXIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: toxic_detect module not available: {e}")
    TOXIC_AVAILABLE = False

try:
    from detects.spam_detect import analyze_text as spam_analyze

    SPAM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: spam_detect module not available: {e}")
    SPAM_AVAILABLE = False

try:
    from classifiers.comment_classifier import (
        classify_comment_with_verification,
        build_prompt,
    )

    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: comment_classifier module not available: {e}")
    CLASSIFIER_AVAILABLE = False

# RAG system will be imported lazily when needed
RAG_AVAILABLE = True


class ComprehensivePipeline:
    """Main pipeline orchestrator for comment processing."""

    def __init__(self):
        """Initialize the pipeline with all components."""
        self.async_processor = AsyncOpenaiProcessor(max_concurrent=10)
        self.rag_system = None
        self._initialize_rag()

    def _initialize_rag(self):
        """Initialize RAG system for answer generation."""
        try:
            print("🔄 Initializing RAG system...")
            from .rag import SimpleCleanRAG

            self.rag_system = SimpleCleanRAG()
            print("✅ RAG system initialized successfully")
        except ImportError as e:
            print(f"⚠️ RAG system import failed: {e}")
            self.rag_system = None
        except Exception as e:
            print(f"❌ RAG system initialization failed: {e}")
            import traceback

            traceback.print_exc()
            self.rag_system = None

    def detect_toxicity_batch_internal(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect toxicity using enhanced RuBERT model with advanced profanity detection.
        Features: Levenshtein distance, substring matching, character normalization.
        """
        print("🔍 Using enhanced RuBERT toxicity detection...")

        # Get enhanced RuBERT results (includes Levenshtein distance, substring matching, etc.)
        rubert_results = []
        if TOXIC_AVAILABLE:
            try:
                rubert_results = detect_toxicity_batch(texts, threshold=0.6)
            except Exception as e:
                print(f"❌ Enhanced RuBERT toxicity detection failed: {e}")
                rubert_results = [
                    {
                        "is_toxic": False,
                        "toxic_score": 0.0,
                        "reason": "Enhanced RuBERT detection failed",
                    }
                    for _ in texts
                ]
        else:
            rubert_results = [
                {
                    "is_toxic": False,
                    "toxic_score": 0.0,
                    "reason": "Enhanced RuBERT not available",
                }
                for _ in texts
            ]

        # Use RuBERT results directly (no more CleanTalk weighting needed)
        combined_results = []
        for rb_result in rubert_results:
            combined_results.append(
                {
                    "is_toxic_rubert": rb_result["is_toxic"],
                    "toxic_score_rubert": rb_result["toxic_score"],
                    "toxic_reason_rubert": rb_result["reason"],
                    "is_toxic": rb_result["is_toxic"],
                    "toxic_score": rb_result["toxic_score"],
                    "reason": rb_result["reason"],
                }
            )

        return combined_results

    async def spam_detection_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect spam for a batch of texts asynchronously.
        Returns list of dicts with spam detection results.
        """
        # Create async requests for spam detection
        requests = []
        for i, text in enumerate(texts):
            # Use the spam detection prompt from spam_detect.py
            messages = [
                {
                    "role": "system",
                    "content": """You are a careful text classifier and translator.

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
}""",
                },
                {
                    "role": "user",
                    "content": f"""INPUT TEXT:
{text}

REQUIREMENTS:
- Detect language (ISO 639-1).
- Translate to Russian literally (no paraphrase).
- Classify spam (true/false) with confidence 0..1.
- Score relevance to Altel/Tele2 in [0..1].
- Return ONLY JSON with keys: language, translated_russian, is_spam_llm, llm_confidence, relevance_altel_tele2, spam_reason_llm.""",
                },
            ]

            requests.append(
                OpenAIRequest(
                    messages=messages,
                    model="gpt-4o-mini-2024-07-18",
                    temperature=0.1,
                    request_id=f"spam_{i}",
                )
            )

        # Process all requests asynchronously
        responses = await self.async_processor.process_requests(requests)

        results = []
        for response in responses:
            if response.success:
                try:
                    # Debug: check if response content is empty or malformed
                    if not response.content or response.content.strip() == "":
                        print(
                            f"⚠️ Empty response content for spam detection request {response.request_id}"
                        )
                        raise json.JSONDecodeError("Empty response", "", 0)

                    # Clean response content (remove markdown formatting if present)
                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = (
                            content.replace("```json", "").replace("```", "").strip()
                        )
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()

                    data = json.loads(content)

                    # Get HF model results for the translated Russian text
                    translated_text = data.get("translated_russian", "")
                    hf_is_spam, hf_confidence, hf_reason = (
                        False,
                        0.0,
                        "HF model not available",
                    )

                    if translated_text and SPAM_AVAILABLE:
                        try:
                            from detects.spam_detect import hf_spam_check_ru

                            hf_is_spam, hf_confidence, hf_reason = hf_spam_check_ru(
                                translated_text
                            )
                        except Exception as e:
                            hf_reason = f"HF model error: {e}"

                    # Final spam decision with 80/20 weight (LLM/HF)
                    llm_is_spam = data.get("is_spam_llm", False)
                    llm_confidence = data.get("llm_confidence", 0.0)

                    # Convert confidence to spam probability (0.0 = not spam, 1.0 = spam)
                    llm_spam_probability = (
                        llm_confidence if llm_is_spam else (1.0 - llm_confidence)
                    )
                    hf_spam_probability = (
                        hf_confidence if hf_is_spam else (1.0 - hf_confidence)
                    )

                    # Weighted final decision: 80% LLM, 20% HF
                    weighted_spam_probability = (0.8 * llm_spam_probability) + (
                        0.2 * hf_spam_probability
                    )
                    final_is_spam = weighted_spam_probability > 0.5

                    results.append(
                        {
                            "language": data.get("language", "unknown"),
                            "is_spam_llm": llm_is_spam,
                            "spam_confidence_llm": llm_confidence,
                            "spam_probability_llm": llm_spam_probability,
                            "is_spam_hf": hf_is_spam,
                            "spam_confidence_hf": hf_confidence,
                            "spam_probability_hf": hf_spam_probability,
                            "spam_reason_hf": hf_reason,
                            "is_spam": final_is_spam,
                            "spam_probability_final": weighted_spam_probability,
                            "relevance_altel_tele2": data.get(
                                "relevance_altel_tele2", 0.0
                            ),
                            "spam_reason": f"Weighted decision (80% LLM, 20% HF): {weighted_spam_probability:.3f}",
                        }
                    )
                except (json.JSONDecodeError, ValueError) as e:
                    print(
                        f"❌ JSON decode error for spam detection request {response.request_id}: {e}"
                    )
                    print(
                        f"   Raw response content: '{response.content[:200]}...' (truncated)"
                    )
                    results.append(
                        {
                            "language": "unknown",
                            "is_spam_llm": False,
                            "spam_confidence_llm": 0.0,
                            "spam_probability_llm": 0.0,
                            "is_spam_hf": False,
                            "spam_confidence_hf": 0.0,
                            "spam_probability_hf": 0.0,
                            "spam_reason_hf": "No HF analysis due to LLM error",
                            "is_spam": False,
                            "spam_probability_final": 0.0,
                            "relevance_altel_tele2": 0.0,
                            "spam_reason": "JSON parse error",
                        }
                    )
            else:
                results.append(
                    {
                        "language": "unknown",
                        "is_spam_llm": False,
                        "spam_confidence_llm": 0.0,
                        "spam_probability_llm": 0.0,
                        "is_spam_hf": False,
                        "spam_confidence_hf": 0.0,
                        "spam_probability_hf": 0.0,
                        "spam_reason_hf": "No HF analysis due to LLM error",
                        "is_spam": False,
                        "spam_probability_final": 0.0,
                        "relevance_altel_tele2": 0.0,
                        "spam_reason": f"API error: {response.error}",
                    }
                )

        return results

    async def classification_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify comments for tone and category asynchronously with both LLM and RuBERT scores.
        Returns list of dicts with classification results from both models.
        """
        # Create async requests for LLM classification
        requests = []
        for i, text in enumerate(texts):
            # Use the classification prompt from comment_classifier.py
            from classifiers.comment_classifier import build_prompt

            messages = [{"role": "system", "content": build_prompt(text)}]

            requests.append(
                OpenAIRequest(
                    messages=messages,
                    model="gpt-4o-mini-2024-07-18",
                    temperature=0.0,
                    request_id=f"classify_{i}",
                )
            )

        # Process all LLM requests asynchronously
        responses = await self.async_processor.process_requests(requests)

        # Get LLM results
        llm_results = []
        for response in responses:
            if response.success:
                try:
                    # Debug: check if response content is empty or malformed
                    if not response.content or response.content.strip() == "":
                        print(
                            f"⚠️ Empty response content for classification request {response.request_id}"
                        )
                        raise json.JSONDecodeError("Empty response", "", 0)

                    # Clean response content (remove markdown formatting if present)
                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = (
                            content.replace("```json", "").replace("```", "").strip()
                        )
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()

                    data = json.loads(content)

                    # Validate required fields
                    if not isinstance(data, dict):
                        raise ValueError("Response is not a JSON object")

                    llm_results.append(
                        {
                            "label_llm": data.get("label", "unknown"),
                            "tone_llm": data.get("tone", "neutral"),
                            "confidence_llm": data.get("confidence", 0.0),
                        }
                    )
                except (json.JSONDecodeError, ValueError) as e:
                    print(
                        f"❌ JSON decode error for LLM classification request {response.request_id}: {e}"
                    )
                    print(
                        f"   Raw response content: '{response.content[:200]}...' (truncated)"
                    )
                    llm_results.append(
                        {
                            "label_llm": "unknown",
                            "tone_llm": "neutral",
                            "confidence_llm": 0.0,
                        }
                    )
            else:
                llm_results.append(
                    {
                        "label_llm": "unknown",
                        "tone_llm": "neutral",
                        "confidence_llm": 0.0,
                    }
                )

        # Get RuBERT results if available
        rubert_results = []
        if CLASSIFIER_AVAILABLE:
            try:
                from classifiers.comment_classifier import classify_with_rubert

                for text in texts:
                    try:
                        rubert_result = classify_with_rubert(text)
                        rubert_results.append(
                            {
                                "label_rubert": rubert_result.get("label", "unknown"),
                                "tone_rubert": rubert_result.get("tone", "neutral"),
                                "confidence_rubert": rubert_result.get(
                                    "confidence", 0.0
                                ),
                            }
                        )
                    except Exception as e:
                        print(f"❌ RuBERT classification error: {e}")
                        rubert_results.append(
                            {
                                "label_rubert": "unknown",
                                "tone_rubert": "neutral",
                                "confidence_rubert": 0.0,
                            }
                        )
            except ImportError:
                print("⚠️ RuBERT classification not available")
                rubert_results = [
                    {
                        "label_rubert": "unknown",
                        "tone_rubert": "neutral",
                        "confidence_rubert": 0.0,
                    }
                    for _ in texts
                ]
        else:
            rubert_results = [
                {
                    "label_rubert": "unknown",
                    "tone_rubert": "neutral",
                    "confidence_rubert": 0.0,
                }
                for _ in texts
            ]

        # Combine results and apply reconciliation logic
        combined_results = []
        for llm_res, rubert_res in zip(llm_results, rubert_results):
            # Combine both results
            combined = {**llm_res, **rubert_res}

            # Determine final labels based on agreement/confidence
            label_agree = llm_res["label_llm"] == rubert_res["label_rubert"]
            tone_agree = llm_res["tone_llm"] == rubert_res["tone_rubert"]

            # Final decision logic (prefer LLM but boost confidence if both agree)
            if label_agree and tone_agree:
                final_label = llm_res["label_llm"]
                final_tone = llm_res["tone_llm"]
                final_confidence = max(
                    llm_res["confidence_llm"], rubert_res["confidence_rubert"]
                )
                final_reason = "Both models agree"
            elif (
                rubert_res["confidence_rubert"] > llm_res["confidence_llm"] + 0.2
            ):  # RuBERT significantly more confident
                final_label = rubert_res["label_rubert"]
                final_tone = rubert_res["tone_rubert"]
                final_confidence = rubert_res["confidence_rubert"]
                final_reason = "RuBERT override (higher confidence)"
            else:
                final_label = llm_res["label_llm"]
                final_tone = llm_res["tone_llm"]
                final_confidence = (
                    max(0.0, llm_res["confidence_llm"] - 0.1)
                    if not label_agree or not tone_agree
                    else llm_res["confidence_llm"]
                )
                final_reason = "LLM primary" + (
                    " (disagreement penalty)"
                    if not label_agree or not tone_agree
                    else ""
                )

            combined.update(
                {
                    "label": final_label,
                    "tone": final_tone,
                    "confidence": final_confidence,
                }
            )

            combined_results.append(combined)

        return combined_results

    async def generate_answers_batch(
        self,
        texts: List[str],
        indices: List[int],
        spam_results: List[Dict[str, Any]],
        task_progress: dict,
    ) -> Dict[int, str]:
        """
        Generate answers for eligible texts (non-toxic, non-spam, relevant).
        Returns dict mapping original indices to generated answers.
        """
        if not self.rag_system:
            print("⚠️ RAG system not available, skipping answer generation")
            return {}

        # Create async requests for answer generation
        requests = []
        index_mapping = {}

        for i, (text, original_idx) in enumerate(zip(texts, indices)):
            # Use RAG system to get context first (synchronously for now)
            try:
                rag_result = self.rag_system.query(text)
                context = "\n".join(
                    [chunk["text"] for chunk in rag_result.get("chunks", [])]
                )

                # Use LLM-detected language from spam detection results
                # Find the corresponding spam result for this text
                original_text_index = indices[i]
                detected_language = spam_results[original_text_index]["language"]

                # Map ISO codes to full language names for the prompt
                language_mapping = {
                    "kk": "Kazakh",
                    "ru": "Russian",
                    "en": "English",
                    "unknown": "Russian",  # Default fallback
                }

                language = language_mapping.get(detected_language, "Russian")
                print(
                    f"   🌐 Language detected by LLM for '{text[:50]}...': {detected_language} -> {language}"
                )

                messages = [
                    {
                        "role": "user",
                        "content": f"""You are a helpful customer service assistant for Altel and Tele2 telecom companies in Kazakhstan.

User question: "{text}"

Context information:
{context}

CRITICAL INSTRUCTION: You MUST respond in {language} language ONLY. Do NOT mix languages or translate the question. Match the exact language of the user's question.

Please provide a helpful answer based on the context information provided above. Follow these guidelines:
- Respond STRICTLY in {language} language - this is mandatory
- If the question is in Kazakh, answer ONLY in Kazakh
- If the question is in Russian, answer ONLY in Russian  
- Do NOT translate or explain in another language
- Be polite and professional
- Be concise but complete
- Use at most one emoji at the end
- Do NOT use any text formatting like bold, italic, or markdown
- Write in plain text only
- Answer based on the context provided - if relevant information is in the context, use it to answer

Answer in {language}:""",
                    }
                ]

                requests.append(
                    OpenAIRequest(
                        messages=messages,
                        model="gpt-4o-mini-2024-07-18",
                        temperature=0.3,
                        request_id=f"answer_{i}",
                    )
                )
                index_mapping[i] = original_text_index

            except Exception as e:
                print(
                    f"❌ Error preparing answer request for index {original_text_index}: {e}"
                )

        if not requests:
            return {}

        # Process all requests asynchronously
        responses = await self.async_processor.process_requests(requests)

        answers = {}
        for i, response in enumerate(responses):
            original_idx = index_mapping.get(i)
            if original_idx is not None:
                if response.success:
                    answer = response.content.strip()
                    answers[original_idx] = answer
                else:
                    answers[original_idx] = (
                        "I apologize, but I'm unable to process your question right now."
                    )

        return answers

    async def process_comments(
        self, df: pd.DataFrame, task_progress: dict
    ) -> pd.DataFrame:
        """
        Process all comments through the complete pipeline.
        """
        texts = df["text"].tolist()
        total_steps = len(texts) * 4  # 4 main processing steps per comment
        task_progress["total_steps"] = total_steps
        task_progress["current_step"] = 0

        print(f"🔄 Processing {len(texts)} comments...")

        # Step 1: Toxicity Detection (synchronous - uses transformers)
        print("1️⃣ Detecting toxicity...")
        toxicity_results = self.detect_toxicity_batch_internal(texts)
        task_progress["current_step"] += len(texts)
        task_progress["progress"] = task_progress["current_step"]

        # Step 2: Spam Detection (asynchronous)
        print("2️⃣ Detecting spam...")
        spam_results = await self.spam_detection_batch(texts)
        task_progress["current_step"] += len(texts)
        task_progress["progress"] = task_progress["current_step"]

        # Step 3: Comment Classification (asynchronous)
        print("3️⃣ Classifying comments...")
        classification_results = await self.classification_batch(texts)
        task_progress["current_step"] += len(texts)
        task_progress["progress"] = task_progress["current_step"]

        # Step 4: Answer Generation (asynchronous, only for eligible comments)
        print("4️⃣ Generating answers...")
        eligible_indices = []
        eligible_texts = []

        for i, (tox, spam, cls) in enumerate(
            zip(toxicity_results, spam_results, classification_results)
        ):
            # Only generate answers for non-toxic, non-spam, relevant comments
            # Relaxed criteria: allow low relevance (>= 0.3) and prioritize non-toxic, non-spam
            if (
                not tox["is_toxic"]
                and not spam["is_spam"]
                and spam["relevance_altel_tele2"] >= 0.3
            ):
                eligible_indices.append(i)
                eligible_texts.append(texts[i])
                print(
                    f"   ✅ Comment {i+1} eligible: toxic={tox['is_toxic']}, spam={spam['is_spam']}, relevance={spam['relevance_altel_tele2']:.2f}"
                )

        print(
            f"   📝 Generating answers for {len(eligible_texts)} eligible comments..."
        )
        answers = await self.generate_answers_batch(
            eligible_texts, eligible_indices, spam_results, task_progress
        )

        # Update progress after answer generation
        task_progress["current_step"] += len(texts)
        task_progress["progress"] = task_progress["current_step"]

        # Step 5: Combine all results
        print("5️⃣ Combining results...")

        # Add new columns to dataframe
        df_result = df.copy()

        # Toxicity columns - Enhanced RuBERT results
        df_result["is_toxic_rubert"] = [r["is_toxic_rubert"] for r in toxicity_results]
        df_result["toxic_score_rubert"] = [
            r["toxic_score_rubert"] for r in toxicity_results
        ]
        df_result["toxic_reason_rubert"] = [
            r["toxic_reason_rubert"] for r in toxicity_results
        ]

        # Final toxicity results (enhanced RuBERT with advanced profanity detection)
        df_result["is_toxic"] = [r["is_toxic"] for r in toxicity_results]
        df_result["toxic_score"] = [r["toxic_score"] for r in toxicity_results]
        df_result["toxic_reason"] = [r["reason"] for r in toxicity_results]

        # Spam columns - basic info
        df_result["language"] = [r["language"] for r in spam_results]
        df_result["relevance_altel_tele2"] = [
            r["relevance_altel_tele2"] for r in spam_results
        ]

        # Spam columns - LLM results
        df_result["is_spam_llm"] = [r["is_spam_llm"] for r in spam_results]
        df_result["spam_confidence_llm"] = [
            r["spam_confidence_llm"] for r in spam_results
        ]
        df_result["spam_probability_llm"] = [
            r["spam_probability_llm"] for r in spam_results
        ]

        # Spam columns - RUSpam/spam_deberta_v4 results
        df_result["is_spam_RUSpam/spam_deberta_v4"] = [
            r["is_spam_hf"] for r in spam_results
        ]
        df_result["spam_confidence_RUSpam/spam_deberta_v4"] = [
            r["spam_confidence_hf"] for r in spam_results
        ]
        df_result["spam_probability_RUSpam/spam_deberta_v4"] = [
            r["spam_probability_hf"] for r in spam_results
        ]
        df_result["spam_reason_RUSpam/spam_deberta_v4"] = [
            r["spam_reason_hf"] for r in spam_results
        ]

        # Final spam results (combined with 80/20 weighting)
        df_result["is_spam"] = [r["is_spam"] for r in spam_results]
        df_result["spam_probability_final"] = [
            r["spam_probability_final"] for r in spam_results
        ]
        df_result["spam_reason"] = [r["spam_reason"] for r in spam_results]

        # Classification columns - LLM results
        df_result["label_llm"] = [r["label_llm"] for r in classification_results]
        df_result["tone_llm"] = [r["tone_llm"] for r in classification_results]
        df_result["confidence_llm"] = [
            r["confidence_llm"] for r in classification_results
        ]

        # Classification columns - RuBERT results
        df_result["label_rubert"] = [r["label_rubert"] for r in classification_results]
        df_result["tone_rubert"] = [r["tone_rubert"] for r in classification_results]
        df_result["confidence_rubert"] = [
            r["confidence_rubert"] for r in classification_results
        ]

        # Final classification results (combined)
        df_result["label"] = [r["label"] for r in classification_results]
        df_result["tone"] = [r["tone"] for r in classification_results]
        df_result["classification_confidence"] = [
            r["confidence"] for r in classification_results
        ]

        # Answer column
        df_result["generated_answer"] = ""
        for idx, answer in answers.items():
            df_result.at[idx, "generated_answer"] = answer

        print("✅ Pipeline processing complete!")
        return df_result


async def main():
    """Main function to run the pipeline."""
    # Load the comments
    # try:
    #     df = pd.read_excel("comments.xlsx")
    #     print(f"📊 Loaded {len(df)} comments from comments.xlsx")
    # except Exception as e:
    #     print(f"❌ Error loading comments.xlsx: {e}")
    #     return

    # Initialize and run pipeline
    # pipeline = ComprehensivePipeline()

    # Process comments
    # result_df = await pipeline.process_comments(df)

    # Save results
    # output_file = "comments_processed.xlsx"
    # try:
    #     result_df.to_excel(output_file, index=False)
    #     print(f"💾 Results saved to {output_file}")

    #     # Print summary
    #     print("\n📊 Processing Summary:")
    #     print(f"Total comments: {len(result_df)}")
    #     print(f"Toxic comments: {result_df['is_toxic'].sum()}")
    #     print(f"Spam comments: {result_df['is_spam'].sum()}")
    #     print(f"Relevant comments: {(result_df['relevance_altel_tele2'] > 0).sum()}")
    #     print(f"Generated answers: {(result_df['generated_answer'] != '').sum()}")

    #     # Show label distribution
    #     print(f"\nLabel distribution:")
    #     print(result_df["label"].value_counts())

    #     print(f"\nTone distribution:")
    #     print(result_df["tone"].value_counts())

    # except Exception as e:
    #     print(f"❌ Error saving results: {e}")
    pass


if __name__ == "__main__":
    asyncio.run(main())
