import torch
import torch.nn as nn
import torch.nn.functional as nnf
from .CRF_layer import CRF
from .weight_drop import WeightDrop
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

class BiLSTM_basic(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 device,
                 bidirectional,
                 batch_first,
                 num_layers,
                 RNNcell,
                 IsDropout,
                 IsDropConnect,
                 dropout=0.4,
                 weightdrop=0.4):
        super(BiLSTM_basic, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        self.bidiectional = bidirectional
        self.IsDropout = IsDropout
        self.dropout = dropout
        self.weight_drop = WeightDrop

        if RNNcell == "GRU":
            self.rnn_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size // 2 if bidirectional else hidden_size,
                                    num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional,
                                    dropout=dropout if num_layers>1 and IsDropout else 0)
            self.RNN_type = "GRU"
        elif RNNcell == "LSTM":
            self.rnn_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size // 2 if bidirectional else hidden_size,
                                      num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional,
                                     dropout=dropout if num_layers>1 and IsDropout else 0)
            self.RNN_type = "LSTM"
        else:
            raise ValueError("{} is unknown RNN cell type, please input GRU/LSTM".format(RNNcell))

        # apply weight-drop to our model
        if IsDropConnect:
            weight_drop_list = [name for name, _ in self.rnn_layer.named_parameters() if "weight_hh" in name]
            self.rnn_layer = WeightDrop(self.rnn_layer, weight_drop_list, dropout=weightdrop)

    def init_hidden(self, batch_size, hidden_dim):
        if self.RNN_type == "LSTM":
            return (torch.randn(self.num_layers * self.num_directions, batch_size, hidden_dim).to(self.device),
                    torch.randn(self.num_layers * self.num_directions, batch_size, hidden_dim).to(self.device))
        else:
                return torch.randn(self.num_layers * self.num_directions, batch_size, hidden_dim).to(self.device)

    def predict(self, *inputs, **kwargs):
        raise NotImplementedError


class BiLSTM_model(BiLSTM_basic):
    def __init__(self, tag_num, freeze_embed, embedding_weights, IsEmbDropout, Embdropout, *args, **kwargs):
        super(BiLSTM_model, self).__init__(*args, **kwargs)
        self.num_labels = tag_num
        self.lookup_table = nn.Embedding.from_pretrained(embedding_weights, freeze=freeze_embed)

        # apply embedding dropout for embedding
        self.embed_drop = nn.Dropout(p=IsEmbDropout if Embdropout else 0)
        self.classifier = nn.Linear(self.hidden_dim, tag_num)

    def forward(self, sents_id, gt_seq_tag_list=None, mask=None, reduction='sum'):
        """
        :param sents_id: [B * L]
        :param elmo_embedding: [B * L * 1024]
        :param gt_seq_tag_list: [B * L]
        :param mask: [B * L] or None
        :return: loss
        """
        batch_size = sents_id.shape[0]

        embeddings = self.lookup_table(sents_id)  # [B * L * embedding_size]
        embeddings = self.embed_drop(embeddings)

        if self.RNN_type == "GRU":
            h_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
            repre_layer, _ = self.rnn_layer(embeddings, h_0)
            logits = self.classifier(repre_layer)
        else:
            h1_0, c1_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
            repre_layer, _ = self.rnn_layer(embeddings, (h1_0, c1_0))
            logits = self.classifier(repre_layer)  # [B * L * tag_num]

        if gt_seq_tag_list is not None:
            loss_fct = nn.CrossEntropyLoss()
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = gt_seq_tag_list.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), gt_seq_tag_list.view(-1))
            return loss, logits
        else:
            return logits

    def predict(self, sents_id, mask=None):
        """
        :param sents_id:
        :param elmo_embedding: [B * L ]
        :param mask: [B * L]
        :return: List of list containing the best tag sequence for each batch.
        """
        batch_size = sents_id.shape[0]
        logits = self.forward(sents_id)  # [B * L * tag_num]
        ids = torch.argmax(logits, dim=-1)  # [B * L]
        seq_ends = mask.long().sum(dim=1) - 1
        ids = ids.to('cpu')
        seq_ends = seq_ends.to('cpu')
        pred_id_list = []
        for row in range(batch_size):
            pred_id_list.append(ids[row][:seq_ends[row]].tolist())
        return pred_id_list


class BiLSTM_CRF_model(BiLSTM_basic):
    def __init__(self, tag_num, freeze_embed, embedding_weights, IsEmbDropout, Embdropout, *args, **kwargs):
        super(BiLSTM_CRF_model, self).__init__(*args, **kwargs)
        self.num_labels = tag_num
        self.lookup_table = nn.Embedding.from_pretrained(embedding_weights, freeze=freeze_embed)

        # apply embedding dropout for embedding
        self.embed_drop = nn.Dropout(p=IsEmbDropout if Embdropout else 0)

        self.lookup_table = nn.Embedding.from_pretrained(embedding_weights, freeze=freeze_embed)
        self.hidden2tag = nn.Linear(self.hidden_dim, tag_num)
        self.CRF_layer = CRF(tag_num, self.batch_first)

    def get_feats(self, sents_id):
        """
        First, get the representation of words after passing through Bi-RNN and then concat with elmo
        elmo embedding to enrich its representation
        :param sents_id: [B * L]
        :param elmo_embedding: [B * L * 1024]
        :return: [B * L * T] if batch_first is true, else [L * B * T]
        """
        batch_size = sents_id.shape[0]

        embeddings = self.lookup_table(sents_id)  # [B * L * embedding_size]
        embeddings = self.embed_drop(embeddings)
        if self.RNN_type == "GRU":
            h_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
            repre_layer, _ = self.rnn_layer(embeddings, h_0)
            feats = self.hidden2tag(repre_layer)
        else:
            h1_0, c1_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
            repre_layer, _ = self.rnn_layer(embeddings, (h1_0, c1_0))
            feats = self.hidden2tag(repre_layer)
        return feats

    def forward(self, sents_id, gt_seq_tag_list, mask=None, reduction='sum'):
        """
        :param sents_id: [B * L]
        :param elmo_embedding: [B * L * 1024]
        :param gt_seq_tag_list: [B * L]
        :param mask: [B * L] or None
        :return: loss
        """
        feats = self.get_feats(sents_id)
        if not self.batch_first:
            gt_seq_tag_list = torch.transpose(gt_seq_tag_list, 0, 1).contiguous()  # [L * B]
            if mask is not None:
                mask = torch.transpose(mask, 0, 1).contiguous()
        loss = self.CRF_layer(feats, gt_seq_tag_list, mask, reduction)
        return loss

    def predict(self, sents_id, mask=None):
        """
        :param sents_id:
        :param elmo_embedding: [B * L ]
        :param mask: [B * L]
        :return: List of list containing the best tag sequence for each batch.
        """
        feats = self.get_feats(sents_id)
        predict_list = self.CRF_layer.decode(feats, mask)
        return predict_list


class Char_BiLSTM_CRF_Model(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 device,
                 bidirectional,
                 batch_first,
                 num_layers,
                 tag_num,
                 freeze_embed,
                 embedding_weights,
                 alphabet_size,
                 char_mode="CNN",
                 char_embed_dim=25,
                 char_out_dim=25,
                 IsDropout=True,
                 dropout=0.4,
                 IsDropConnect=True,
                 weightdrop=0.4
                 ):
        super(Char_BiLSTM_CRF_Model, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        self.bidiectional = bidirectional
        self.num_labels = tag_num
        self.lookup_table = nn.Embedding.from_pretrained(embedding_weights, freeze=freeze_embed)
        self.char_lookup_table = nn.Embedding(alphabet_size, char_embed_dim)
        self.char_lookup_table.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, char_embed_dim)))
        self.char_mode = char_mode

        if char_mode == "LSTM":
            self.char_lstm = nn.LSTM(char_embed_dim, char_out_dim // 2, batch_first=True, num_layers=1, bidirectional=True)
            char_out_dim = char_out_dim // 2
            self.rnn_layer = nn.LSTM(input_size=input_size+char_out_dim * 2,
                                     hidden_size=hidden_size // 2 if bidirectional else hidden_size,
                                     num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional,
                                     dropout=dropout if num_layers>1 and IsDropout else 0)
        elif char_mode == "CNN":
            self.char_cnn = nn.Conv1d(in_channels=char_embed_dim, out_channels=char_out_dim, kernel_size=3, padding=1)
            self.rnn_layer = nn.LSTM(input_size=input_size + char_out_dim,
                                     hidden_size=hidden_size // 2 if bidirectional else hidden_size,
                                     num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional,
                                     dropout=dropout if num_layers>1 and IsDropout else 0)

        # apply embedding dropout for embedding
        self.embed_drop = nn.Dropout(p=dropout if IsDropout else 0)

        # apply weight-drop to our model
        if IsDropConnect:
            weight_drop_list = [name for name, _ in self.rnn_layer.named_parameters() if "weight_hh" in name]
            self.rnn_layer = WeightDrop(self.rnn_layer, weight_drop_list, dropout=weightdrop)

        self.hidden2tag = nn.Linear(hidden_size, tag_num)
        self.CRF_layer = CRF(tag_num, self.batch_first)

    def init_hidden(self, batch_size, hidden_dim):
        return (torch.randn(self.num_layers * self.num_directions, batch_size, hidden_dim).to(self.device),
                torch.randn(self.num_layers * self.num_directions, batch_size, hidden_dim).to(self.device))

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def char_conv(self, char_embedding_matrix):
        """char_embedding_matrix: [L * LW * CE]
        """
        char_embedding_matrix = char_embedding_matrix.transpose(1, 2)  # [L * CE * LW], transpose for conv1d
        cnn_out = self.char_cnn(char_embedding_matrix)  # [L, out_channels, L]
        cnn_out = nnf.max_pool1d(cnn_out, cnn_out.shape[2])  # [L, out_channels, 1]
        return cnn_out.squeeze(-1)

    def char_rnn(self, char_embedding_matrix, word_lengths):
        """
        char_embedding_matrix: [L * LW * CE], word_length: [L]
        """
        sentence_length = char_embedding_matrix.shape[0]
        lengths_dict = [(idx, length) for idx, length in enumerate(word_lengths)]
        length_sorted = sorted(lengths_dict, key=lambda a: a[1], reverse=True)
        ordered_length = [length[1] for length in length_sorted]
        idx_origin_order = [length[0] for length in length_sorted]
        idx_origin_order_dict = dict([(idx, origin_idx) for origin_idx, idx in enumerate(idx_origin_order)])
        idx_back_to_origin_order = [idx_origin_order_dict[i] for i in range(sentence_length)]

        char_embedding_matrix = char_embedding_matrix[idx_origin_order]
        pack_input = pack_padded_sequence(char_embedding_matrix, ordered_length, batch_first=True)

        char_hidden = None
        rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_hidden = char_hidden[0].transpose(0, 1).contiguous().view(sentence_length, -1)  # [L * CE], with ordered length
        return char_hidden[idx_back_to_origin_order]


    def get_feats(self, word_input_ids, char_input_ids, word_lengths):
        """
        B: batch_size, L: sentence length, LW: word length, CE: char embedding size, WE: word embedding size
        :param word_input_ids(tensor): [B * L]
        :param char_input_ids(tensor): [B * L * LW]
        :param word_lengths: [B * L]
        :return:
        """
        batch_size = word_input_ids.shape[0]
        char_embedding_list = []
        for one_batch, lengths in zip(char_input_ids, word_lengths):
            char_embedding = self.char_lookup_table(one_batch)  # [L * LW * CE]
            if self.char_mode == "CNN":
                cnn_out = self.char_conv(char_embedding)  # [L * CE]
                char_embedding_list.append(cnn_out)
            else:
                rnn_out = self.char_rnn(char_embedding, lengths)
                char_embedding_list.append(rnn_out)
        char_embedding_for_all = torch.stack(char_embedding_list, dim=0)  # [B * L * CE]

        word_embedding = self.lookup_table(word_input_ids)  # [B * L * WE]
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = torch.cat((word_embedding, char_embedding_for_all), dim=-1)

        h1_0, c1_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
        repre_layer, _ = self.rnn_layer(word_embedding, (h1_0, c1_0))
        feats = self.hidden2tag(repre_layer)
        return feats

    def forward(self, word_input_ids, char_input_ids, gt_seq_tag_list, word_lengths, mask=None, reduction='sum'):
        feats = self.get_feats(word_input_ids, char_input_ids, word_lengths)
        if not self.batch_first:
            gt_seq_tag_list = torch.transpose(gt_seq_tag_list, 0, 1).contiguous()  # [L * B]
            if mask is not None:
                mask = torch.transpose(mask, 0, 1).contiguous()
        loss = self.CRF_layer(feats, gt_seq_tag_list, mask, reduction)
        return loss

    def predict(self, word_input_ids, char_input_ids, word_lengths, mask=None):
        feats = self.get_feats(word_input_ids, char_input_ids, word_lengths)
        predict_list = self.CRF_layer.decode(feats, mask)
        return predict_list





