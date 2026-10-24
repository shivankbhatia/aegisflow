from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)