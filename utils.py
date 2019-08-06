import random
import numpy as np
from tqdm import tqdm
import copy
def data_reader(data_file_name, data_type, length_cut_off=150, bert_tokenizer=None):
    """
    :param when using bert for NER, use bert_tokenizer for tokenization
    :param data_file_name: file name of dataset
    :param length_cut_off: if sentence is too long, we split it into 150 length
    :return:
    """
    def length_limited(sent, cut_off):
        res = []
        segment_num = int(len(sent) / cut_off)
        remain = len(sent) % cut_off
        for i in range(segment_num):
            res.append(sent[i*cut_off:(i+1)*cut_off])
        if remain != 0:
            res.append(sent[-remain:])
        return res

    data_reader = list()
    vocab = set()

    sent_with_label = []
    with open(data_file_name, mode='r', encoding="utf-8") as f:
        for No, line in enumerate(f):
            if line.strip() == "":
                if len(sent_with_label) == 0:
                    continue
                tmp_result = length_limited(sent_with_label, length_cut_off)
                data_reader += tmp_result
                sent_with_label.clear()
            else:
                if data_type in ["weibo", "people_daily"]:
                    tag = line.split()
                else:
                    tag = line.split("\t")
                if bert_tokenizer is None:
                    vocab.add(tag[0].strip())
                    sent_with_label.append((tag[0].strip(), tag[1].strip()))
                else:
                    words, label = tag[0].strip(), tag[1].strip()
                    tokens = bert_tokenizer.tokenize(words)
                    if len(tokens) == 0:
                        print(words)
                        print(No)
                        print(label)
                    if label.startswith('O'):
                        for i in range(len(tokens)):
                            sent_with_label.append((tokens[i], 'O'))
                    elif label.startswith('B'):
                        sent_with_label.append((tokens[0], label))
                        for i in range(1, len(tokens)):
                            sent_with_label.append((tokens[i], label.replace('B', 'I')))
                    else:
                        for i in range(len(tokens)):
                            sent_with_label.append((tokens[i], label))

    return data_reader, vocab


def get_embedding_weights_and_vocab(vocab, pretrained_embedding_file, embedding_dim):
    """
    :param vocab: set()
    :param pretrained_embedding_file: file name of pretrained embedding weights
    :return:
    """
    def read_in_chunks(file_path, chunk_size):
        with open(file_path, mode='r', encoding="utf-8") as file_object:
            while True:
                chunk_data = file_object.readlines(chunk_size)
                if not chunk_data:
                    break
                yield chunk_data

    embedding_weights = np.random.randn(len(vocab)+1, embedding_dim)
    embedding_weights[0] = np.zeros(embedding_dim, dtype=np.float32)
    data_vocab = {'<pad>': 0}
    for No, key in enumerate(vocab):
        data_vocab[key] = No+1

    with open(pretrained_embedding_file, mode='r', encoding="utf-8") as f:
        for line in tqdm(f):
        # for line in chunk:
            word = line.split()[0]
            if word in vocab and len(line.split()[1:]) == embedding_dim:
                embedding_weights[data_vocab[word]] = np.array(line.split()[1:], dtype=np.float32)

    return embedding_weights, data_vocab


def padding_batch_of_sequence(sents_texts, total_length, pad_value="<pad>"):
    """
    this function is mainly for making the inputs label_ids padded so that we could convert a batch
    of label ids into tensor format
    :param sents_texts: [B,]
    :param total_length: the max length of sentence
    :param pad_value: <pad>
    :return: sents_after_padding: [B * MAX_LENGTH]
    """
    new_sents_texts = copy.deepcopy(sents_texts)
    for i in range(len(sents_texts)):
        if len(new_sents_texts[i]) < total_length:
            new_sents_texts[i] += [pad_value] * (total_length-len(new_sents_texts[i]))
    return new_sents_texts


def sentence_padding(sent_id, max_length):
    if len(sent_id) >= max_length:
        return sent_id[:max_length], [1]*max_length
    else:
        return sent_id + [0] * (max_length - len(sent_id)), [1]*len(sent_id) + [0]*(max_length - len(sent_id))


def convert_sents_to_ids(sents, vocab):
    """
    convert the words in sents into ids for later word-to-vec,
    :param sents: [B,]
    :param vocab: dict
    :return: sents into ids with padding, dimension is [B * L], L means max_length. and mask
    """
    sents_id = []
    sents_mask = []
    max_length = len(sents[-1])
    for sent in sents:
        # print(sent)
        sent_id = [vocab[word[0]] for word in sent]
        sent_id_padding, sent_mask = sentence_padding(sent_id, max_length)
        sents_id.append(sent_id_padding)
        sents_mask.append(sent_mask)
    return sents_id, sents_mask


def convert_labels_to_ids(labels_list, tag_to_idx):
    return [[tag_to_idx[label] for label in label_list] for label_list in labels_list]


def train_test_split(corpus, proportion=0.1):
    sample_num = int(len(corpus)*proportion)
    # print("num:", sample_num)
    val_id = random.sample(range(len(corpus)), sample_num)
    train_corpus, dev_corpus = [], []
    for i in range(len(corpus)):
        if i in val_id: dev_corpus.append(corpus[i])
        else: train_corpus.append(corpus[i])
    return train_corpus, dev_corpus


tag_to_idx_msra = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'B-PER': 3, 'I-PER': 4, 'B-LOC': 5, 'I-LOC': 6}

tag_to_idx_weibo = {'O': 0, 'B-ORG.NAM': 1, 'I-ORG.NAM': 2, 'B-PER.NAM': 3, 'I-PER.NAM': 4, 'B-LOC.NAM': 5, 'I-LOC.NAM': 6,
                    'B-ORG.NOM': 7, 'I-ORG.NOM': 8, 'B-PER.NOM': 9, 'I-PER.NOM': 10, 'B-LOC.NOM': 11, 'I-LOC.NOM': 12,
                    'B-GPE.NAM': 13, 'I-GPE.NAM': 14, 'B-GPE.NOM': 15, 'I-GPE.NOM': 16}

tag_to_idx_people = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'B-PER': 3, 'I-PER': 4, 'B-LOC': 5, 'I-LOC': 6}

tag_to_idx_finance = {'O': 0, 'B-CPY': 1, 'I-CPY': 2, 'B-PER': 3, 'I-PER': 4, 'B-MISC': 5, 'I-MISC': 6, 'B-BRAND': 7, 'I-BRAND': 8}

tag_to_idx_jiaming = {'O': 0, 'B-BOD': 1, 'I-BOD': 2, 'B-DIS': 3, 'I-DIS': 4, 'B-SYM': 5, 'I-SYM': 6, 'B-TRE': 7, 'I-TRE': 8, 'B-CHE': 9, 'I-CHE': 10}

