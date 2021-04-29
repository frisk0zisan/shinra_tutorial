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

def decode_output(labels, infos, is_text=False, is_ene=False, is_title=False):
    chunks = []
    for label, info in zip(labels, infos):
        label = ["O"] + label + ["O"]
        for idx in range(1, len(label)):
            if is_chunk_end(label[idx-1], label[idx]):
                assert len(chunks) > 0
                _, attribute = split_tag(label[idx])
                chunks[-1]["text_offset"]["end"] = {"line_id": int(info['line_id']), "offset": int(info['text_offset'][idx-2][1])}
                if is_text is True: # add text
                    chunks[-1]["text_offset"]["text"] = info['text']

            if is_chunk_start(label[idx-1], label[idx]):
                _, attribute = split_tag(label[idx])
                if is_ene is True: # add ENE
                    chunks.append({"page_id": info['page_id'], "attribute": attribute, "ENE": info['ENE'], 
                               "text_offset": {"start": {"line_id": int(info['line_id']), "offset": int(info['text_offset'][idx-1][0])}}})
                elif is_title is True: # add title
                    chunks.append({"page_id": info['page_id'],"attribute": attribute, "title":info['title'], 
                               "text_offset": {"start": {"line_id": int(info['line_id']), "offset": int(info['text_offset'][idx-1][0])}}})
                else :
                    chunks.append({"page_id": info['page_id'], "attribute": attribute,
                               "text_offset": {"start": {"line_id": int(info['line_id']), "offset": int(info['text_offset'][idx-1][0])}}})

    return chunks

def print_shinra_format(chunks, path):
    chunks = [json.dumps(c, ensure_ascii=False) for c in chunks]
    with open(path, 'w') as f:
        f.write('\n'.join(chunks))

def add_title_to_infos(infos, path):
    for info in infos:
            with open(path, "r") as f:
                for line in f:
                    line = line.rstrip()
                    if not line:
                        continue
                    line = json.loads(line)
                    if line['page_id'] == info['page_id']:
                        info['title'] = line['title']
                        break
    return infos

def add_ENE_to_infos(infos, path):
    for info in infos:
            with open(path, "r") as f:
                for line in f:
                    line = line.rstrip()
                    if not line:
                        continue
                    line = json.loads(line)
                    if line['page_id'] == info['page_id']:
                        info['ENE'] = line['ENE']
                        break
    return infos
