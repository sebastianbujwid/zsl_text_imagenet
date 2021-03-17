def split_with_overlap(text, max_length, overlap_window_length, tokenize_func=lambda x: x.split()):
    assert overlap_window_length < max_length

    tokens = tokenize_func(text)
    if len(tokens) < max_length:
        return [' '.join(tokens)]

    split_text = []
    i = 0
    while (i + overlap_window_length) <= len(tokens):
        part = tokens[i:i + max_length]
        split_text.append(' '.join(part))
        i += max_length - overlap_window_length

    return split_text
