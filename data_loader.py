from utils import convert_sents_to_ids, convert_labels_to_ids
import random
import re

class Instance_char_lstm(object):
    def __init__(self, sents_ids, char_ids, sents_tag_ids, sents_masks, word_lengths):
        self.sents_ids = sents_ids
        self.sents_tag_ids = sents_tag_ids
        self.sents_masks = sents_masks
        self.char_ids = char_ids
        self.word_lengths = word_lengths

    def __getitem__(self):
        return self.sents_ids, self.char_ids, self.sents_tag_ids, self.sents_masks, self.word_lengths


class Dataloader_basic(object):
    def __init__(self, data_reader, batch_size):
        self.data_reader = data_reader
        self.data_loader = []
        self.data_reader.sort(key=lambda element: len(element))
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data_reader)

    def steps(self):
        return self.__len__() // self.batch_size

    def Random_loading(self):
        random.shuffle(self.data_loader)
        for data_batch in self.data_loader:
            yield data_batch

    def sequential_loading(self):
        for data_batch in self.data_loader:
            yield data_batch


class Char_lstm_Dataloader(Dataloader_basic):
    def sents_padding(self, sents):
        sents_masks = []
        max_length = len(sents[-1])
        sents_padding = []
        for sent in sents:
            text = [element[0] for element in sent]
            sents_masks.append(len(sent)*[1]+(max_length-len(sent))*[0])
            sents_padding.append(text+["<pad>"]*(max_length-len(sent)))
        return sents_padding, sents_masks

    def convert_char_to_ids(self, batch_text, max_word_length, char_vocab, wubi_dict, char_mode="CNN"):
        batch_char_ids, batch_word_length = [], []
        for text in batch_text:
            char_ids, word_lengths = [], []
            for word in text:
                if word == "<pad>":
                    char_ids.append([0])
                    word_lengths += [1]
                else:
                    if re.match(r"[\u4e00-\u9fa5]", word) is not None:
                        wubi = wubi_dict[word]
                        char_ids.append([char_vocab[character] for character in wubi])
                        word_lengths += [len(wubi)]
                    else:
                        char_ids.append([1])
                        word_lengths += [1]

            # padding to max_word_length
            for i in range(len(char_ids)):
                if word_lengths[i] > max_word_length:
                    char_ids[i] = char_ids[i][:max_word_length]
                else:
                    if char_mode == "LSTM":
                        char_ids[i] += [0]*(max_word_length-word_lengths[i])
                    else:
                        left_pad = (max_word_length-word_lengths[i]) // 2
                        right_pad = max_word_length-word_lengths[i]-left_pad
                        char_ids[i] = [0]*left_pad+char_ids[i]+[0]*right_pad
            batch_char_ids.append(char_ids)
            batch_word_length.append(word_lengths)
        return batch_char_ids, batch_word_length


    def create_one_batch(self, vocab, tag_to_idx, max_word_length, char_vocab, wubi_token, char_mode="CNN"):
        """
        :param vocab: a dictionary which map a word into id
        :param tag_to_idx: a dictionary which map a label into label id
        :param max_word_length: the max length for a word. It's been using for padding when getting char ids
        :param char_vocab: a dictionary which map a character into id
        :param wubi_token: a toolkit which convert word in wubi
        :param char_mode: CNN/LSTM
        """
        whole_steps = self.steps()
        for i in range(whole_steps):
            sents_batch = self.data_reader[i*self.batch_size:(i+1)*self.batch_size]
            sent_with_pad, sents_mask = self.sents_padding(sents_batch)
            batch_char_ids, batch_word_lengths = self.convert_char_to_ids(sent_with_pad, max_word_length, char_vocab, wubi_token, char_mode)

            sents_ids = [[vocab[word] for word in sent] for sent in sent_with_pad]
            sents_label = [[element[1] for element in sent] for sent in sents_batch]
            sents_label_id = convert_labels_to_ids(sents_label, tag_to_idx)
            one_batch = Instance_char_lstm(sents_ids, batch_char_ids, sents_label_id, sents_mask, batch_word_lengths)
            self.data_loader.append(one_batch)