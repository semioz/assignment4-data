def passes_gopher_quality_filters(text: str) -> bool:
    words = text.split()
    if not 50 <= len(words) <= 100_000:
        return False

    mean_word_length = sum(map(len, words)) / len(words)
    if not 3 <= mean_word_length <= 10:
        return False

    lines = text.splitlines()
    if lines and sum(line.rstrip().endswith("...") for line in lines) / len(lines) > 0.3:
        return False

    return sum(any(char.isalpha() for char in word) for word in words) / len(words) >= 0.8
