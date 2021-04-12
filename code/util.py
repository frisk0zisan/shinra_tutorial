import re
import json

def split_tag(label):
    if label == "O":
        return ("O", None)
    return label.split("-", maxsplit=1)

def is_chunk_end(prev_label, label):
    prefix, label = split_tag(label)
    prev_prefix, prev_label = split_tag(prev_label)

    if prev_prefix == "O":
        return False

    if prefix == "O":
        return prev_prefix != "O"

    if label != prev_label:
        return True

    return prefix == "B"

def is_chunk_start(prev_label, label):
    prefix, label = split_tag(label)
    prev_prefix, prev_label = split_tag(prev_label)

    if prefix == "O":
        return False

    if prev_prefix == "O":
        return prefix != "O"

    if label != prev_label:
        return True

    return prefix == "B"

def decode_output(labels, infos):
    chunks = []
    for label, info in zip(labels, infos):
        label = ["O"] + label + ["O"]
        for idx in range(1, len(label)):
            if is_chunk_end(label[idx-1], label[idx]):
                assert len(chunks) > 0
                _, attribute = split_tag(label[idx])
                chunks[-1]["text_offset"]["end"] = {"line_id": info['line_id'], "offset": info['text_offset'][idx-1][1]}
            if is_chunk_start(label[idx-1], label[idx]):
                _, attribute = split_tag(label[idx])
                chunks.append({"page_id": info['page_id'], "attribute": attribute,
                               "text_offset": {"start": {"line_id": info['line_id'], "offset": info['text_offset'][idx-1][0]}}})

    return chunks

def print_shinra_format(chunks, path):
    chunks = [json.dumps(c, ensure_ascii=False) for c in chunks]
    with open(path, 'w') as f:
        f.write('\n'.join(chunks))

