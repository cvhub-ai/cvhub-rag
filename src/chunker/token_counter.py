from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TokenCounter:
    def __init__(self) -> None:
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            "BAAI/bge-m3"
        )

    def count(self, text: str) -> int:
        return len(
            self._tokenizer.encode(
                text,
                add_special_tokens=True,
            )
        )

    @property
    def special_tokens_count(self) -> int:
        return self._tokenizer.num_special_tokens_to_add(pair=False)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
