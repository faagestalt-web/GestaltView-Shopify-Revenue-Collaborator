from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Dict, Any

import re


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9']*", text.lower())


class ManifestIndex:
    """Lightweight manifest index for compressing a corpus into key insights."""

    def ingest(self, documents: Iterable[str], loom_targets: Iterable[str] | None = None) -> Dict[str, Any]:
        docs = [d for d in documents if d]
        tokens = _tokenize("\n".join(docs))
        counts = Counter(tokens)
        top_terms = [term for term, _ in counts.most_common(20)]

        sentences = [s.strip() for s in re.split(r"[.!?]+", "\n".join(docs)) if s.strip()]
        sample_sentences = sentences[:5]

        return {
            "document_count": len(docs),
            "total_tokens": len(tokens),
            "top_terms": top_terms,
            "loom_targets": list(loom_targets or []),
            "sample_sentences": sample_sentences,
        }

    def find_emergent_patterns(self, documents: Iterable[str], query_focus: str) -> List[Dict[str, Any]]:
        """Return simple clusters of themes related to the query focus."""
        docs = [d for d in documents if d]
        tokens = _tokenize("\n".join(docs))
        counts = Counter(tokens)
        query_tokens = set(_tokenize(query_focus))

        relevant = [term for term, _ in counts.most_common(30) if not query_tokens or term in query_tokens]
        if not relevant:
            relevant = [term for term, _ in counts.most_common(10)]

        clusters = []
        for idx, term in enumerate(relevant[:5], start=1):
            clusters.append(
                {
                    "label": f"Cluster {idx}: {term.title()}",
                    "keywords": [term],
                    "score": counts[term],
                }
            )
        return clusters
