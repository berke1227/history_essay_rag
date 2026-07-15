"""Katmanlar arası paylaşılan, saf veri modelleri (davranış içermez)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """Bir makaleden çıkarılmış tek bir metin parçası."""

    id: str
    text: str
    source_file: str
    chunk_index: int


@dataclass(frozen=True)
class RetrievedChunk:
    """Sorguya karşı getirilen bir parça ve benzerlik skoru."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class VerificationResult:
    """Doğrulayıcı LLM çağrısının sonucu."""

    passed: bool
    feedback: str


@dataclass(frozen=True)
class Answer:
    """Kullanıcıya dönecek nihai cevap ve üretim süreci hakkında bilgi."""

    text: str
    grounded: bool
    attempts: int
    sources: list[str] = field(default_factory=list)
