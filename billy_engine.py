from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Any

from ..models import BucketDropCapture, TapestryResponse, PLKProfile
from .manifest_index import ManifestIndex
from ..context_ingestion import calculate_plk


class BillyEngine:
    """Consciousness-serving synthesis engine for a specific client."""

    def __init__(self, client_id: str, plk_profile: PLKProfile | None, corpus_docs: List[str]):
        self.client_id = client_id
        self.plk = plk_profile
        self.corpus = corpus_docs
        self.context_spine: Dict[str, Any] = {"bucket_drops": [], "loom_threads": []}
        self.manifest_index = ManifestIndex()

    def detect_mood(self, text: str) -> str:
        lower = text.lower()
        if any(word in lower for word in ["overwhelmed", "stuck", "panic", "storm"]):
            return "overwhelmed"
        if any(word in lower for word in ["excited", "spark", "ignite", "flow"]):
            return "energized"
        if any(word in lower for word in ["curious", "wonder", "explore"]):
            return "curious"
        return "steady"

    def find_loom_connections(self, text: str) -> List[str]:
        tokens = set(text.lower().split())
        connections: List[str] = []
        for thread in self.context_spine.get("loom_threads", []):
            overlap = tokens.intersection(set(thread.get("tokens", [])))
            if overlap:
                connections.append(thread.get("label", "previous thread"))
        return connections

    def archive_with_context(self, capture: BucketDropCapture, threads: List[str]) -> None:
        self.context_spine.setdefault("bucket_drops", []).append(capture.dict())
        self.context_spine.setdefault("loom_threads", []).append(
            {
                "label": f"drop-{len(self.context_spine['bucket_drops'])}",
                "tokens": capture.raw_input.lower().split(),
                "connections": threads,
            }
        )

    def generate_with_plk_mirror(self, clusters: List[Dict[str, Any]], plk: PLKProfile | None) -> str:
        metaphors = plk.signature_metaphors if plk else []
        metaphor_line = f"Weaving through your metaphors ({', '.join(metaphors[:2])})" if metaphors else "Weaving through your language"
        cluster_lines = "\n".join([f"- {c['label']}: {', '.join(c['keywords'])}" for c in clusters])
        return (
            f"{metaphor_line}, here is what surfaced:\n"
            f"{cluster_lines}\n"
            "Let me know which thread feels most alive, and we can deepen it."
        )

    def process_bucket_drop(self, spontaneous_input: str) -> BucketDropCapture:
        capture = BucketDropCapture(
            raw_input=spontaneous_input,
            timestamp=datetime.utcnow().isoformat(),
            mood_signature=self.detect_mood(spontaneous_input),
            loom_threads=[],
            resonance_score=None,
        )

        threads = self.find_loom_connections(spontaneous_input)
        capture.loom_threads = threads

        self.archive_with_context(capture, threads)
        return capture

    def synthesize_tapestry(self, query: str) -> TapestryResponse:
        clusters = self.manifest_index.find_emergent_patterns(self.corpus, query_focus=query)
        narrative = self.generate_with_plk_mirror(clusters=clusters, plk=self.plk)
        return TapestryResponse(query=query, narrative=narrative, clusters=clusters)

    def continuous_learning(self, interaction_history: List[Dict[str, Any]]) -> PLKProfile:
        texts = [entry.get("content", "") for entry in interaction_history if entry.get("content")]
        combined = self.corpus + texts
        plk_dict = calculate_plk(combined)
        self.plk = PLKProfile.model_validate(plk_dict)
        return self.plk
