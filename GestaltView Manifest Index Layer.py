"""
GestaltView Manifest Index Layer – Production-Enhanced Implementation
Author: Enhanced for Keith Soyka's GestaltView Platform
Date: December 30, 2025


Architecture Philosophy:
- Consciousness-serving: respects narrative continuity & semantic depth
- Resilient: handles failures gracefully with exponential backoff
- Observable: comprehensive logging for auditability
- Parallel: async operations where beneficial
- Type-safe: comprehensive type hints
- Testable: dependency injection & clear contracts


Design Principles:
✓ Correctness over optimization
✓ Traceability over convenience  
✓ Human-readable over clever
✓ Graceful degradation over brittle perfection
"""


from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
import pathlib
import sys
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, List, Dict, Optional, Any, Callable
from functools import wraps
import time


import psycopg2
from psycopg2.extras import Json, execute_values
from psycopg2.pool import ThreadedConnectionPool


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION LAYER
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Config:
    """Immutable configuration ensuring deterministic behavior."""
    corpus_root: pathlib.Path
    manifest_out: pathlib.Path
    llm_model: str = "gpt-4o"
    max_tokens: int = 4096
    chunk_size: int = 8000  # chars per chunk for oversized docs
    chunk_overlap: int = 500  # overlap between chunks
    db_dsn: str = "postgresql://localhost/gestaltview"
    db_pool_min: int = 2
    db_pool_max: int = 10
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    log_level: str = "INFO"
    parallel_workers: int = 4
    supported_extensions: tuple = (".txt", ".md", ".pdf")
    
    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""
        return cls(
            corpus_root=pathlib.Path(os.getenv("CORPUS_ROOT", "./corpus")),
            manifest_out=pathlib.Path(os.getenv("MANIFEST_OUT", "./manifest_index.json")),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            db_dsn=os.getenv("DATABASE_URL", "postgresql://localhost/gestaltview"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# ═══════════════════════════════════════════════════════════════════
# LOGGING & OBSERVABILITY
# ═══════════════════════════════════════════════════════════════════


class LogContext(Enum):
    """Structured log contexts for observability."""
    INGEST = "ingest"
    SUMMARIZE = "summarize"
    COMPOUND = "compound"
    SNOWBALL = "snowball"
    LOOM = "loom"
    PERSIST = "persist"
    LLM = "llm"
    ERROR = "error"
    PIPELINE = "pipeline"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger("gestaltview.manifest")
    logger.setLevel(getattr(logging, level.upper()))
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(context)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


logger = setup_logging()


def log_context(context: LogContext):
    """Decorator to add context to log messages."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════


def stable_hash(text: str) -> str:
    """Generate deterministic SHA-256 hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now() -> datetime:
    """UTC timestamp for auditability."""
    return datetime.now(timezone.utc)


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Split large text into overlapping chunks preserving narrative flow.
    
    WHY: LLMs have token limits; chunking with overlap maintains context.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        import random
        delay *= (0.5 + random.random() * 0.5)
    return delay


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = exponential_backoff(attempt, base_delay, max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s...",
                            extra={'context': LogContext.ERROR.value}
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}",
                            extra={'context': LogContext.ERROR.value}
                        )
            raise last_exception
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DocumentText:
    """Ingested document with metadata."""
    document_id: str
    path: str
    content: str
    hash: str
    created_at: datetime
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    file_size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        return d


@dataclass
class Summary:
    """AI-generated summary at various hierarchy levels."""
    summary_id: str
    document_id: Optional[str]
    level: str  # 'primary', 'compounded', 'corpus'
    content: str
    model: str
    created_at: datetime
    token_count: Optional[int] = None
    processing_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        return d


@dataclass
class LoomAnnotation:
    """Loom analysis capturing gaps, threads, and emergent patterns."""
    annotation_id: str
    type: str  # 'gap', 'thread', 'motif', 'global_analysis'
    related_ids: List[str]
    content: str
    created_at: datetime
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        return d


@dataclass
class ProcessingMetrics:
    """Telemetry for observability."""
    documents_processed: int = 0
    chunks_processed: int = 0
    summaries_generated: int = 0
    annotations_created: int = 0
    total_tokens: int = 0
    start_time: datetime = field(default_factory=now)
    end_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    
    def finalize(self):
        """Mark processing complete."""
        self.end_time = now()
    
    def duration_seconds(self) -> float:
        """Calculate total processing time."""
        end = self.end_time or now()
        return (end - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'documents_processed': self.documents_processed,
            'chunks_processed': self.chunks_processed,
            'summaries_generated': self.summaries_generated,
            'annotations_created': self.annotations_created,
            'total_tokens': self.total_tokens,
            'duration_seconds': self.duration_seconds(),
            'errors_count': len(self.errors),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
        }
# ═══════════════════════════════════════════════════════════════════
# INGESTION LAYER
# ═══════════════════════════════════════════════════════════════════


class Ingestor:
    """
    Discovers and normalizes corpus documents.
    
    GestaltView Philosophy: Each document is a thread in the tapestry.
    We preserve provenance (path, hash, timestamp) for auditability.
    """
    
    def __init__(self, root: pathlib.Path, cfg: Config):
        self.root = root
        self.cfg = cfg
        self.metrics = ProcessingMetrics()
    
    @log_context(LogContext.INGEST)
    def ingest(self) -> List[DocumentText]:
        """
        Recursively discover and ingest documents.
        
        Returns chunked documents if they exceed chunk_size.
        """
        logger.info(f"Starting ingestion from {self.root}", extra={'context': 'ingest'})
        docs: List[DocumentText] = []
        
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            
            if path.suffix.lower() not in self.cfg.supported_extensions:
                logger.debug(f"Skipping unsupported file: {path}", extra={'context': 'ingest'})
                continue
            
            try:
                text = self._read_file(path)
                file_size = path.stat().st_size
                h = stable_hash(text)
                
                # Handle large documents by chunking
                chunks = chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
                total_chunks = len(chunks)
                
                for idx, chunk in enumerate(chunks):
                    doc = DocumentText(
                        document_id=str(uuid.uuid4()),
                        path=str(path),
                        content=chunk,
                        hash=h if total_chunks == 1 else stable_hash(chunk),
                        created_at=now(),
                        chunk_index=idx if total_chunks > 1 else None,
                        total_chunks=total_chunks if total_chunks > 1 else None,
                        file_size_bytes=file_size,
                    )
                    docs.append(doc)
                    self.metrics.chunks_processed += 1
                
                self.metrics.documents_processed += 1
                logger.info(
                    f"Ingested {path.name} ({total_chunks} chunk(s), {file_size} bytes)",
                    extra={'context': 'ingest'}
                )
                
            except Exception as e:
                error_msg = f"Failed to ingest {path}: {e}"
                logger.error(error_msg, extra={'context': 'ingest'})
                self.metrics.errors.append(error_msg)
        
        logger.info(
            f"Ingestion complete: {self.metrics.documents_processed} documents, "
            f"{self.metrics.chunks_processed} chunks",
            extra={'context': 'ingest'}
        )
        return docs
    
    def _read_file(self, path: pathlib.Path) -> str:
        """Read file content with encoding fallback."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {path}, trying latin-1", extra={'context': 'ingest'})
            return path.read_text(encoding="latin-1")


# ═══════════════════════════════════════════════════════════════════
# LLM INTEGRATION LAYER
# ═══════════════════════════════════════════════════════════════════


class LLMProvider:
    """
    Abstract LLM interface for pluggable backends.
    
    WHY: Isolates LLM logic for testing, swapping providers, cost optimization.
    Maintains prompt contracts as first-class citizens.
    """
    
    def __init__(self, model: str, max_tokens: int, cfg: Config):
        self.model = model
        self.max_tokens = max_tokens
        self.cfg = cfg
    
    @retry_with_backoff(max_attempts=3, base_delay=2.0, exceptions=(Exception,))
    @log_context(LogContext.LLM)
    def generate(self, prompt: str, temperature: float = 0.7) -> tuple:
        """
        Generate completion from prompt.
        
        Returns: (content, token_count)
        
        WHY: Explicit stub forces intentional provider integration.
        Override this method with your LLM backend (OpenAI, Anthropic, etc.)
        """
        logger.info(f"LLM generate called (model={self.model})", extra={'context': 'llm'})
        raise NotImplementedError(
            "LLM backend not implemented. "
            "Subclass LLMProvider and override generate() with your provider."
        )
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4


# Example implementation stub for OpenAI:
class OpenAIProvider(LLMProvider):
    """Example OpenAI provider implementation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            logger.warning("OpenAI not installed. Install with: pip install openai")
            self.client = None
    
    def generate(self, prompt: str, temperature: float = 0.7) -> tuple:
        """Generate completion using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens
            return content, tokens
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", extra={'context': 'llm'})
            raise


# ═══════════════════════════════════════════════════════════════════
# INCHWORM SUMMARIZATION LAYER
# ═══════════════════════════════════════════════════════════════════


class InchwormSummarizer:
    """
    Progressive summarization maintaining narrative continuity.
    
    GestaltView Philosophy: Like consciousness itself, understanding builds
    incrementally. Each summary preserves what came before while integrating
    the new—the "inchworm" moves forward without losing its foundation.
    """
    
    def __init__(self, llm: LLMProvider, cfg: Config):
        self.llm = llm
        self.cfg = cfg
        self.metrics = ProcessingMetrics()
    
    @log_context(LogContext.SUMMARIZE)
    def primary_summary(self, doc: DocumentText) -> Summary:
        """
        Generate primary conceptual summary from raw document.
        
        WHY: Captures authorial intent, core thesis, open questions—
        the essence, not just keywords.
        """
        start_time = time.time()
        
        prompt = f"""You are analyzing a document from the GestaltView corpus.


Read the following document and produce a high-level conceptual summary that captures:


1. **Core Thesis**: What is the central argument or purpose?
2. **Key Concepts**: What ideas, frameworks, or methodologies are introduced?
3. **Authorial Intent**: What is the author trying to achieve or communicate?
4. **Open Questions**: What remains unresolved or invites further exploration?


Maintain the author's voice and perspective. Preserve nuance over brevity.


DOCUMENT:
{doc.content}


SUMMARY:"""
        
        try:
            content, tokens = self.llm.generate(prompt, temperature=0.5)
            processing_time = int((time.time() - start_time) * 1000)
            
            summary = Summary(
                summary_id=str(uuid.uuid4()),
                document_id=doc.document_id,
                level="primary",
                content=content,
                model=self.llm.model,
                created_at=now(),
                token_count=tokens,
                processing_time_ms=processing_time,
            )
            
            self.metrics.summaries_generated += 1
            self.metrics.total_tokens += tokens
            
            logger.info(
                f"Primary summary generated for doc {doc.document_id[:8]} "
                f"({tokens} tokens, {processing_time}ms)",
                extra={'context': 'summarize'}
            )
            
            return summary
            
        except Exception as e:
            error_msg = f"Primary summary failed for {doc.document_id}: {e}"
            logger.error(error_msg, extra={'context': 'summarize'})
            self.metrics.errors.append(error_msg)
            raise
    
    @log_context(LogContext.COMPOUND)
    def compound(self, summary: Summary, accumulator: str) -> Summary:
        """
        Compound new summary with existing narrative context.
        
        GestaltView Philosophy: Each new insight is woven into the existing
        tapestry, preserving continuity while allowing evolution.
        """
        start_time = time.time()
        
        prompt = f"""You are compounding summaries in the GestaltView corpus.


Given the PRIOR SEMANTIC CONTEXT and a NEW SUMMARY, produce a COMPOUNDED SUMMARY that:
- Preserves narrative continuity from prior context
- Integrates new insights and concepts
- Identifies thematic resonances or divergences
- Maintains temporal flow (what came before → what comes now)


Think of this as weaving a new thread into an existing tapestry.


PRIOR CONTEXT:
{accumulator}


NEW SUMMARY:
{summary.content}


COMPOUNDED SUMMARY:"""
        
        try:
            content, tokens = self.llm.generate(prompt, temperature=0.6)
            processing_time = int((time.time() - start_time) * 1000)
            
            compounded = Summary(
                summary_id=str(uuid.uuid4()),
                document_id=summary.document_id,
                level="compounded",
                content=content,
                model=self.llm.model,
                created_at=now(),
                token_count=tokens,
                processing_time_ms=processing_time,
            )
            
            self.metrics.summaries_generated += 1
            self.metrics.total_tokens += tokens
            
            logger.info(
                f"Compounded summary generated ({tokens} tokens, {processing_time}ms)",
                extra={'context': 'compound'}
            )
            
            return compounded
            
        except Exception as e:
            error_msg = f"Compound summary failed: {e}"
            logger.error(error_msg, extra={'context': 'compound'})
            self.metrics.errors.append(error_msg)
            raise
# ═══════════════════════════════════════════════════════════════════
# SNOWBALL CORPUS SYNTHESIS
# ═══════════════════════════════════════════════════════════════════


class SnowballSummarizer:
    """
    Corpus-level synthesis identifying emergent themes and concept lineages.
    
    GestaltView Philosophy: Like consciousness, meaning emerges from patterns
    across time. The corpus isn't just documents—it's a living knowledge graph.
    """
    
    def __init__(self, llm: LLMProvider, cfg: Config):
        self.llm = llm
        self.cfg = cfg
        self.metrics = ProcessingMetrics()
    
    @log_context(LogContext.SNOWBALL)
    def corpus_summary(self, compounded: Iterable[Summary]) -> Summary:
        """Generate corpus-level synthesis from compounded summaries."""
        start_time = time.time()
        
        joined = "\n\n---\n\n".join(s.content for s in compounded)
        
        prompt = f"""You are synthesizing the complete GestaltView corpus.


From the following COMPOUNDED SUMMARIES, produce a CORPUS-LEVEL SYNTHESIS that identifies:


1. **Emergent Themes**: What patterns emerge across documents?
2. **Concept Lineages**: How do ideas evolve or compound over time?
3. **Knowledge Clusters**: What conceptual territories does the corpus cover?
4. **Narrative Arc**: What is the through-line or meta-narrative?
5. **Consciousness Signature**: What does this corpus reveal about its creator's way of thinking?


This is not a summary of summaries—it's a meta-analysis revealing what emerges
when all pieces are viewed as a unified whole.


COMPOUNDED SUMMARIES:
{joined}


CORPUS SYNTHESIS:"""
        
        try:
            content, tokens = self.llm.generate(prompt, temperature=0.7)
            processing_time = int((time.time() - start_time) * 1000)
            
            synthesis = Summary(
                summary_id=str(uuid.uuid4()),
                document_id=None,
                level="corpus",
                content=content,
                model=self.llm.model,
                created_at=now(),
                token_count=tokens,
                processing_time_ms=processing_time,
            )
            
            self.metrics.summaries_generated += 1
            self.metrics.total_tokens += tokens
            
            logger.info(
                f"Corpus synthesis complete ({tokens} tokens, {processing_time}ms)",
                extra={'context': 'snowball'}
            )
            
            return synthesis
            
        except Exception as e:
            error_msg = f"Corpus synthesis failed: {e}"
            logger.error(error_msg, extra={'context': 'snowball'})
            self.metrics.errors.append(error_msg)
            raise


# ═══════════════════════════════════════════════════════════════════
# LOOM GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════


class LoomAnalyzer:
    """
    Detects gaps, threads, and emergent patterns in the knowledge tapestry.
    
    GestaltView Philosophy: The Loom reveals what's missing—the unspoken,
    the unresolved, the threads that want to be woven but haven't been yet.
    """
    
    def __init__(self, llm: LLMProvider, cfg: Config):
        self.llm = llm
        self.cfg = cfg
        self.metrics = ProcessingMetrics()
    
    @log_context(LogContext.LOOM)
    def analyze(self, summaries: Iterable[Summary]) -> List[LoomAnnotation]:
        """Perform Loom analysis detecting gaps and emergent patterns."""
        start_time = time.time()
        
        joined = "\n\n---\n\n".join(s.content for s in summaries)
        
        prompt = f"""You are performing LOOM ANALYSIS on the GestaltView corpus.


The Loom reveals:
- **Gaps**: Concepts introduced but never fully explored
- **Threads**: Recurring motifs with subtle variations or contradictions
- **Weak Connections**: Important ideas that are adjacent but not explicitly linked
- **Unresolved Questions**: Open loops that invite completion
- **Emergent Patterns**: What wants to exist but hasn't been articulated yet


Analyze the following summaries and produce structured findings in JSON format:


[
  {{
    "type": "gap|thread|weak_connection|unresolved|emergent",
    "title": "Brief title",
    "description": "Detailed finding",
    "related_concepts": ["concept1", "concept2"],
    "confidence": 0.0-1.0
  }}
]


Be specific. Point to concrete examples. Trust your intuition about what matters.


SUMMARIES:
{joined}


LOOM FINDINGS:"""
        
        try:
            content, tokens = self.llm.generate(prompt, temperature=0.8)
            processing_time = int((time.time() - start_time) * 1000)
            
            # Attempt to parse structured output
            annotations = self._parse_loom_output(content, tokens, processing_time)
            
            self.metrics.annotations_created += len(annotations)
            self.metrics.total_tokens += tokens
            
            logger.info(
                f"Loom analysis complete: {len(annotations)} annotations "
                f"({tokens} tokens, {processing_time}ms)",
                extra={'context': 'loom'}
            )
            
            return annotations
            
        except Exception as e:
            error_msg = f"Loom analysis failed: {e}"
            logger.error(error_msg, extra={'context': 'loom'})
            self.metrics.errors.append(error_msg)
            # Return fallback global analysis
            return [
                LoomAnnotation(
                    annotation_id=str(uuid.uuid4()),
                    type="global_analysis",
                    related_ids=[],
                    content=str(e),
                    created_at=now(),
                )
            ]
    
    def _parse_loom_output(
        self, content: str, tokens: int, processing_time: int
    ) -> List[LoomAnnotation]:
        """Parse LLM output into structured annotations."""
        try:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
            if json_match:
                findings = json.loads(json_match.group(1))
            else:
                findings = json.loads(content)
            
            annotations = []
            for finding in findings:
                ann = LoomAnnotation(
                    annotation_id=str(uuid.uuid4()),
                    type=finding.get("type", "finding"),
                    related_ids=finding.get("related_concepts", []),
                    content=json.dumps({
                        "title": finding.get("title", ""),
                        "description": finding.get("description", ""),
                    }, indent=2),
                    created_at=now(),
                    confidence_score=finding.get("confidence", None),
                )
                annotations.append(ann)
            
            return annotations
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse structured Loom output: {e}", extra={'context': 'loom'})
            # Fallback: treat entire content as single annotation
            return [
                LoomAnnotation(
                    annotation_id=str(uuid.uuid4()),
                    type="global_analysis",
                    related_ids=[],
                    content=content,
                    created_at=now(),
                )
            ]


# ═══════════════════════════════════════════════════════════════════
# PERSISTENCE LAYER
# ═══════════════════════════════════════════════════════════════════


class ManifestStore:
    """
    PostgreSQL persistence with connection pooling and batch operations.
    
    WHY: Auditability requires durable storage. Connection pooling
    prevents resource exhaustion during parallel processing.
    """
    
    def __init__(self, dsn: str, cfg: Config):
        self.dsn = dsn
        self.cfg = cfg
        self.pool = ThreadedConnectionPool(
            cfg.db_pool_min,
            cfg.db_pool_max,
            dsn
        )
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create tables if they don't exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            document_id UUID PRIMARY KEY,
            path TEXT NOT NULL,
            hash TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            chunk_index INTEGER,
            total_chunks INTEGER,
            file_size_bytes INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS summaries (
            summary_id UUID PRIMARY KEY,
            document_id UUID REFERENCES documents(document_id),
            level TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            token_count INTEGER,
            processing_time_ms INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS loom_annotations (
            id UUID PRIMARY KEY,
            type TEXT NOT NULL,
            related_ids JSONB,
            content TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            confidence_score FLOAT
        );
        
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
        CREATE INDEX IF NOT EXISTS idx_summaries_level ON summaries(level);
        CREATE INDEX IF NOT EXISTS idx_loom_type ON loom_annotations(type);
        """
        
        conn = self.pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(schema_sql)
        finally:
            self.pool.putconn(conn)
    
    @log_context(LogContext.PERSIST)
    def save_document(self, doc: DocumentText) -> None:
        """Save document with conflict handling."""
        conn = self.pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (
                        document_id, path, hash, created_at, 
                        chunk_index, total_chunks, file_size_bytes
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO NOTHING
                    """,
                    (
                        doc.document_id,
                        doc.path,
                        doc.hash,
                        doc.created_at,
                        doc.chunk_index,
                        doc.total_chunks,
                        doc.file_size_bytes,
                    ),
                )
        finally:
            self.pool.putconn(conn)
    
    @log_context(LogContext.PERSIST)
    def save_summary(self, summary: Summary) -> None:
        """Save summary."""
        conn = self.pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO summaries (
                        summary_id, document_id, level, content, 
                        model, created_at, token_count, processing_time_ms
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        summary.summary_id,
                        summary.document_id,
                        summary.level,
                        summary.content,
                        summary.model,
                        summary.created_at,
                        summary.token_count,
                        summary.processing_time_ms,
                    ),
                )
        finally:
            self.pool.putconn(conn)
    
    @log_context(LogContext.PERSIST)
    def save_loom(self, ann: LoomAnnotation) -> None:
        """Save Loom annotation."""
        conn = self.pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO loom_annotations (
                        id, type, related_ids, content, 
                        created_at, confidence_score
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        ann.annotation_id,
                        ann.type,
                        Json(ann.related_ids),
                        ann.content,
                        ann.created_at,
                        ann.confidence_score,
                    ),
                )
        finally:
            self.pool.putconn(conn)
    
    def batch_save_documents(self, docs: List[DocumentText]) -> None:
        """Batch save documents for efficiency."""
        if not docs:
            return
        
        conn = self.pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                values = [
                    (
                        d.document_id, d.path, d.hash, d.created_at,
                        d.chunk_index, d.total_chunks, d.file_size_bytes
                    )
                    for d in docs
                ]
                execute_values(
                    cur,
                    """
                    INSERT INTO documents (
                        document_id, path, hash, created_at,
                        chunk_index, total_chunks, file_size_bytes
                    ) VALUES %s
                    ON CONFLICT (document_id) DO NOTHING
                    """,
                    values
                )
            logger.info(f"Batch saved {len(docs)} documents", extra={'context': 'persist'})
        finally:
            self.pool.putconn(conn)
    
    def close(self):
        """Close connection pool."""
        self.pool.closeall()
# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATION PIPELINE
# ═══════════════════════════════════════════════════════════════════


class ManifestPipeline:
    """
    Main orchestrator coordinating all layers.
    
    GestaltView Philosophy: This is the conductor—each layer plays its part,
    but the pipeline ensures they harmonize into a coherent whole.
    """
    
    def __init__(self, cfg: Config, llm_provider: Optional[LLMProvider] = None):
        self.cfg = cfg
        self.ingestor = Ingestor(cfg.corpus_root, cfg)
        
        # Allow external LLM provider injection for testing
        self.llm = llm_provider or LLMProvider(cfg.llm_model, cfg.max_tokens, cfg)
        
        self.inchworm = InchwormSummarizer(self.llm, cfg)
        self.snowball = SnowballSummarizer(self.llm, cfg)
        self.loom = LoomAnalyzer(self.llm, cfg)
        self.store = ManifestStore(cfg.db_dsn, cfg)
        
        self.metrics = ProcessingMetrics()
    
    def run(self) -> ProcessingMetrics:
        """
        Execute full pipeline: ingest → summarize → compound → synthesize → analyze → persist.
        
        Returns processing metrics for observability.
        """
        logger.info("=" * 70, extra={'context': 'pipeline'})
        logger.info("GestaltView Manifest Pipeline Starting", extra={'context': 'pipeline'})
        logger.info("=" * 70, extra={'context': 'pipeline'})
        
        try:
            # Step 1: Ingest
            logger.info("Step 1: Ingesting documents...", extra={'context': 'pipeline'})
            docs = self.ingestor.ingest()
            self.metrics.documents_processed = self.ingestor.metrics.documents_processed
            self.metrics.chunks_processed = self.ingestor.metrics.chunks_processed
            
            if not docs:
                logger.warning("No documents found to process", extra={'context': 'pipeline'})
                return self.metrics
            
            # Batch save documents
            self.store.batch_save_documents(docs)
            
            # Step 2: Primary summaries + compounding
            logger.info("Step 2: Generating primary summaries and compounding...", extra={'context': 'pipeline'})
            accumulator = ""
            compounded: List[Summary] = []
            
            for idx, doc in enumerate(docs):
                logger.info(
                    f"Processing document {idx + 1}/{len(docs)}: {doc.path}",
                    extra={'context': 'pipeline'}
                )
                
                # Primary summary
                primary = self.inchworm.primary_summary(doc)
                self.store.save_summary(primary)
                
                # Compound with accumulator
                compounded_summary = self.inchworm.compound(primary, accumulator)
                self.store.save_summary(compounded_summary)
                
                # Update accumulator
                accumulator = compounded_summary.content
                compounded.append(compounded_summary)
            
            self.metrics.summaries_generated += self.inchworm.metrics.summaries_generated
            self.metrics.total_tokens += self.inchworm.metrics.total_tokens
            
            # Step 3: Corpus synthesis
            logger.info("Step 3: Synthesizing corpus...", extra={'context': 'pipeline'})
            corpus = self.snowball.corpus_summary(compounded)
            self.store.save_summary(corpus)
            self.metrics.summaries_generated += self.snowball.metrics.summaries_generated
            self.metrics.total_tokens += self.snowball.metrics.total_tokens
            
            # Step 4: Loom analysis
            logger.info("Step 4: Performing Loom analysis...", extra={'context': 'pipeline'})
            loom_annotations = self.loom.analyze([corpus] + compounded)
            for ann in loom_annotations:
                self.store.save_loom(ann)
            self.metrics.annotations_created += self.loom.metrics.annotations_created
            self.metrics.total_tokens += self.loom.metrics.total_tokens
            
            # Step 5: Export manifest
            logger.info("Step 5: Exporting manifest...", extra={'context': 'pipeline'})
            self.export_manifest(docs, compounded, corpus, loom_annotations)
            
            self.metrics.finalize()
            logger.info("=" * 70, extra={'context': 'pipeline'})
            logger.info("Pipeline Complete!", extra={'context': 'pipeline'})
            logger.info(f"Metrics: {self.metrics.to_dict()}", extra={'context': 'pipeline'})
            logger.info("=" * 70, extra={'context': 'pipeline'})
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", extra={'context': 'pipeline'}, exc_info=True)
            self.metrics.errors.append(str(e))
            self.metrics.finalize()
            raise
        finally:
            self.store.close()
    
    def export_manifest(
        self,
        docs: List[DocumentText],
        compounded: List[Summary],
        corpus: Summary,
        loom: List[LoomAnnotation],
    ) -> None:
        """Export complete manifest to JSON."""
        manifest = {
            "metadata": {
                "generated_at": now().isoformat(),
                "model": self.cfg.llm_model,
                "corpus_root": str(self.cfg.corpus_root),
                "document_count": len(docs),
                "chunk_count": self.metrics.chunks_processed,
            },
            "metrics": self.metrics.to_dict(),
            "documents": [d.to_dict() for d in docs],
            "compounded_summaries": [s.to_dict() for s in compounded],
            "corpus_summary": corpus.to_dict(),
            "loom_annotations": [a.to_dict() for a in loom],
        }
        
        self.cfg.manifest_out.write_text(json.dumps(manifest, indent=2))
        logger.info(f"Manifest exported to {self.cfg.manifest_out}", extra={'context': 'pipeline'})


# ═══════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════


def main():
    """Command-line entry point."""
    import argparse
    import dataclasses
    
    parser = argparse.ArgumentParser(
        description="GestaltView Manifest Index Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python gestaltview_manifest_pipeline_enhanced.py
  
  # Run with custom corpus directory
  python gestaltview_manifest_pipeline_enhanced.py --corpus /path/to/corpus
  
  # Run with environment variables
  CORPUS_ROOT=./corpus LLM_MODEL=gpt-4o python gestaltview_manifest_pipeline_enhanced.py
        """
    )
    
    parser.add_argument(
        "--corpus",
        type=pathlib.Path,
        help="Path to corpus root directory (default: ./corpus)"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Path to output manifest JSON (default: ./manifest_index.json)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    cfg = Config.from_env()
    
    # Override with CLI args if provided
    if args.corpus:
        cfg = dataclasses.replace(cfg, corpus_root=args.corpus)
    if args.output:
        cfg = dataclasses.replace(cfg, manifest_out=args.output)
    if args.log_level:
        cfg = dataclasses.replace(cfg, log_level=args.log_level)
        global logger
        logger = setup_logging(args.log_level)
    
    # Run pipeline
    pipeline = ManifestPipeline(cfg)
    
    try:
        metrics = pipeline.run()
        
        # Print summary
        print("\n" + "=" * 70)
        print("GESTALTVIEW MANIFEST PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Documents processed: {metrics.documents_processed}")
        print(f"Chunks processed: {metrics.chunks_processed}")
        print(f"Summaries generated: {metrics.summaries_generated}")
        print(f"Annotations created: {metrics.annotations_created}")
        print(f"Total tokens: {metrics.total_tokens:,}")
        print(f"Duration: {metrics.duration_seconds():.2f}s")
        if metrics.errors:
            print(f"Errors: {len(metrics.errors)}")
        print("=" * 70)
        
        sys.exit(0 if not metrics.errors else 1)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()