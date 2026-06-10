from dataclasses import dataclass, field


@dataclass
class Document:
    """A single Stack Exchange post (question or answer) as clean text."""
    post_id: int
    question_id: int          # the owning question's id (== post_id for questions)
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    score: int = 0
    is_accepted: bool = False


@dataclass
class Chunk:
    """A retrievable passage derived from a Document, carrying its provenance."""
    text: str
    post_id: int
    question_id: int
    chunk_index: int = 0      # position of this chunk within its source document


@dataclass
class Retrieved:
    """A chunk returned by retrieval with its relevance score."""
    chunk: Chunk
    score: float
