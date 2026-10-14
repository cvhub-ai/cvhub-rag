from dataclasses import dataclass


@dataclass
class ChunkerConfig:
    target_tokens: int = 350
    max_tokens: int = 512
    overlap_tokens: int = 64
