"""Hybrid summarizer: extractive pre-filtering + abstractive BART."""
from __future__ import annotations

from typing import List, Optional, Dict, Any
import logging
import re

from auto_summarizer.models import get_summarizer  # type: ignore circular import ok

logger = logging.getLogger(__name__)


class HybridSummarizer:
    """Combine simple extractive selection with abstractive BART summarization.

    Steps:
    1. Split text into sentences.
    2. Rank sentences by length (proxy for importance) and select *k* best.
    3. Concatenate selected sentences to form a compressed context.
    4. Generate abstractive summary with BARTSummarizer.
    5. Fallback to concatenated extractive result if abstractive fails.
    """

    def __init__(self, device: Optional[str] = None, k: int = 3):
        self.extractive_k = k
        # Lazy-load BART summarizer via factory (inherits device auto-detection)
        self.abstractive = get_summarizer("bart-large", device=device)

    # ---------------------------------------------------------------------
    # Extractive step
    # ---------------------------------------------------------------------
    _sent_split_re = re.compile(r"(?<=[.!?])\s+")

    def _split_sentences(self, text: str) -> List[str]:
        sentences = self._sent_split_re.split(text.strip())
        # remove empty
        return [s.strip() for s in sentences if s.strip()]

    def _select_key_sentences(self, sentences: List[str]) -> List[str]:
        # Simple heuristic: pick longest sentences (not perfect but fast)
        ranked = sorted(sentences, key=len, reverse=True)
        return ranked[: self.extractive_k]

    # ---------------------------------------------------------------------
    def __call__(self, text: str, **gen_kwargs: Any) -> str:
        if not text.strip():
            return ""

        sentences = self._split_sentences(text)
        if len(sentences) <= self.extractive_k:
            compressed_context = text
        else:
            selected = self._select_key_sentences(sentences)
            compressed_context = " ".join(selected)

        try:
            summary = self.abstractive(
                compressed_context,
                **gen_kwargs,
            )
            if summary:
                return summary
        except Exception as exc:
            logger.error("Hybrid summarizer: abstractive step failed: %s", exc)

        # Fallback to extractive compressed context (truncate to 3 sentences)
        fallback = " ".join(sentences[: self.extractive_k])
        return fallback
