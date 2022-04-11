import spacy


def load_ner(path):
    ner=spacy.load(path)

    label_list = ["O"]
    for label in ner.get_pipe('ner').labels:
        label_list.append('B'+label)
        label_list.append('I'+label)

    tag2id = {t:i for i,t in enumerate(label_list)}
    id2tag = {i:t for i,t in enumerate(label_list)}
    return ner, tag2id, id2tag
