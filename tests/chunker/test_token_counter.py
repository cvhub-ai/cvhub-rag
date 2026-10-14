from src.chunker.token_counter import TokenCounter


def test_token_counter_returns_positive_count():
    token_counter = TokenCounter()

    text = "OCR systems contain text detection and recognition."

    token_count = token_counter.count(text)

    assert isinstance(token_count, int)
    assert token_count > 0


def test_token_counter_longer_text_has_more_tokens():
    token_counter = TokenCounter()

    short_text = "OCR system."
    long_text = "OCR systems contain text detection and text recognition components."

    short_count = token_counter.count(short_text)
    long_count = token_counter.count(long_text)

    assert long_count > short_count


def test_token_counter_empty_text():
    token_counter = TokenCounter()

    token_count = token_counter.count("")

    assert isinstance(token_count, int)
    assert token_count >= 0
