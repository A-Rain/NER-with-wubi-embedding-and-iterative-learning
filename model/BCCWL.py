import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from .CRF_layer import CRF
from .biLSTM_Module import BiLSTM_basic

class Module_auxiliary(BiLSTM_basic):
    def __init__(self, tag_num, *args, **kwargs):
        super(Module_auxiliary, self).__init__(*args, **kwargs)
        self.num_labels = tag_num
        self.classifier = nn.Linear(self.hidden_dim, tag_num)

    def predict(self, logits, masks=None):
        batch_size = logits.shape[0]
        ids = torch.argmax(logits, dim=-1)  # [B * L]
        if masks is None:
            return ids
        else:
            seq_ends = masks.long().sum(dim=1)
            ids = ids.to('cpu')
            seq_ends = seq_ends.to('cpu')
            pred_id_list = []
            for row in range(batch_size):
                pred_id_list.append(ids[row][:seq_ends[row]].tolist())
            return pred_id_list

    def forward(self, embeddings, true_label_list=None, masks=None):
        batch_size = embeddings.shape[0]
        if self.RNN_type == "GRU":
            h_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
            repre_layer, _ = self.rnn_layer(embeddings, h_0)
            logits = self.classifier(repre_layer)
        else:
            h1_0, c1_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
            repre_layer, _ = self.rnn_layer(embeddings, (h1_0, c1_0))
            logits = self.classifier(repre_layer)  # [B * L * tag_num]

        if true_label_list is not None:
            loss_fct = nn.CrossEntropyLoss()
            if masks is not None:
                active_loss = masks.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = true_label_list.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), true_label_list.view(-1))
            return loss, logits
        else:
            return logits


class Module_auxiliary_CRF(BiLSTM_basic):
    def __init__(self, tag_num, *args, **kwargs):
        super(Module_auxiliary_CRF, self).__init__(*args, **kwargs)
        self.hidden2tag = nn.Linear(self.hidden_dim, tag_num)
        self.CRF_layer = CRF(tag_num, self.batch_first)

    def get_feats(self, embeddings):
        batch_size = embeddings.shape[0]
        if self.RNN_type == "GRU":
            h_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
            repre_layer, _ = self.rnn_layer(embeddings, h_0)
            feats = self.hidden2tag(repre_layer)
        else:
            h1_0, c1_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
            repre_layer, _ = self.rnn_layer(embeddings, (h1_0, c1_0))
            feats = self.hidden2tag(repre_layer)
        return feats

    def predict(self, feats, mask=None):
        predict_list = self.CRF_layer.decode(feats, mask)
        return predict_list

    def forward(self, embeddings, true_label_list, mask=None, reduction='sum'):
        feats = self.get_feats(embeddings)
        if not self.batch_first:
            true_label_list = torch.transpose(true_label_list, 0, 1).contiguous()  # [L * B]
            if mask is not None:
                mask = torch.transpose(mask, 0, 1).contiguous()
        loss = self.CRF_layer(feats, true_label_list, mask, reduction)
        predict_list = self.predict(feats)
        return loss, predict_list


class Module_main(BiLSTM_basic):
    def __init__(self, tag_num, *args, **kwargs):
        super(Module_main, self).__init__(*args, **kwargs)
        self.hidden2tag = nn.Linear(self.hidden_dim, tag_num)
        self.CRF_layer = CRF(tag_num, self.batch_first)

    def get_feats(self, embeddings, label_embeddings):
        batch_size = embeddings.shape[0]
        if self.RNN_type == "GRU":
            h_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
            inputs = torch.cat((label_embeddings, embeddings), dim=-1)
            repre_layer, _ = self.rnn_layer(inputs, h_0)
            feats = self.hidden2tag(repre_layer)
        else:
            h1_0, c1_0 = self.init_hidden(batch_size, self.hidden_dim // 2 if self.bidiectional else self.hidden_dim)
            inputs = torch.cat((label_embeddings, embeddings), dim=-1)
            repre_layer, _ = self.rnn_layer(inputs, (h1_0, c1_0))
            feats = self.hidden2tag(repre_layer)
        return feats

    def forward(self, feats, true_label_list, mask=None, reduction='sum'):
        # feats = self.get_feats(embeddings, label_embeddings)
        if not self.batch_first:
            true_label_list = torch.transpose(true_label_list, 0, 1).contiguous()  # [L * B]
            if mask is not None:
                mask = torch.transpose(mask, 0, 1).contiguous()
        loss = self.CRF_layer(feats, true_label_list, mask, reduction)
        return loss

    def predict(self, feats, mask=None):
        # feats = self.get_feats(embeddings, label_embeddings)
        predict_list = self.CRF_layer.decode(feats, mask)
        return predict_list


class New_Model_Wubi_label(nn.Module):
    def __init__(self,
                 freeze_embed,
                 wubi_freeze,
                 embedding_weights,
                 num_labels,
                 alphabet_size,
                 label_embedding_dim,
                 wubi_embedding_dim,
                 wubi_out_dim,
                 wubi_mode,
                 loss_lambda,
                 auxiliary_num,
                 refactor_num,
                 auxiliary_config, main_config):
        super(New_Model_Wubi_label, self).__init__()
        self.num_labels = num_labels
        self.label_embedding_dim = label_embedding_dim
        self.wubi_embedding_dim = wubi_embedding_dim
        self.wubi_out_dim = wubi_out_dim
        self.loss_lambda = loss_lambda
        self.refactor_num = refactor_num
        self.auxiliary_num = auxiliary_num
        self.wubi_mode=wubi_mode

        auxiliary_config["input_size"] += wubi_embedding_dim
        self.Auxiliary_part1 = Module_auxiliary(tag_num=num_labels, **auxiliary_config)
        auxiliary_config["input_size"] += label_embedding_dim
        for i in range(auxiliary_num-1):
            setattr(self, "Auxiliary_part{}".format(i+2), Module_auxiliary(tag_num=num_labels, **auxiliary_config))
        main_config["input_size"] += label_embedding_dim+wubi_embedding_dim
        self.Main = Module_main(tag_num=num_labels, **main_config)
        self.device = self.Main.device

        self.lookup_table = nn.Embedding.from_pretrained(embedding_weights, freeze=freeze_embed)
        self.label_embedding = nn.Embedding(num_labels, label_embedding_dim)
        self.label_embedding.weight.data.copy_(
            torch.from_numpy(self.random_embedding(num_labels, label_embedding_dim)))

        # self.wubi_embedding = nn.Embedding(alphabet_size, wubi_embedding_dim)
        wubi_weights = self.random_embedding(alphabet_size, wubi_embedding_dim)
        wubi_weights[0] = np.zeros([1, wubi_embedding_dim])
        wubi_weights = torch.from_numpy(wubi_weights).float()
        self.wubi_embedding = nn.Embedding.from_pretrained(wubi_weights, freeze=wubi_freeze)

        if wubi_mode == "CNN":
            self.wubi_cnn = nn.Conv1d(in_channels=wubi_embedding_dim, out_channels=wubi_out_dim, kernel_size=3, padding=1)
        else:
           self.wubi_lstm = nn.LSTM(wubi_embedding_dim, wubi_out_dim // 2, batch_first=True, num_layers=1,
                                    bidirectional=True)

        # apply embedding dropout for embedding
        self.embed_drop = nn.Dropout(p=self.Main.dropout if self.Main.IsDropout else 0)

    def random_embedding(self, vocab_num, embedding_dim):
        pretrain_emb = np.empty([vocab_num, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_num):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def wubiEmbed_rnn(self, wubi_ids, word_lengths):
        """
        wubi_ids: [L * LW], word_length: [L]
        """
        wubi_embedding_matrix = self.wubi_embedding(wubi_ids)  # [L * LW * CE]
        sentence_length = wubi_embedding_matrix.shape[0]
        lengths_dict = [(idx, length) for idx, length in enumerate(word_lengths)]
        length_sorted = sorted(lengths_dict, key=lambda a: a[1], reverse=True)
        ordered_length = [length[1] for length in length_sorted]
        idx_origin_order = [length[0] for length in length_sorted]
        idx_origin_order_dict = dict([(idx, origin_idx) for origin_idx, idx in enumerate(idx_origin_order)])
        idx_back_to_origin_order = [idx_origin_order_dict[i] for i in range(sentence_length)]

        wubi_embedding_matrix = wubi_embedding_matrix[idx_origin_order]
        # print(ordered_length)
        # print("\n")
        pack_input = pack_padded_sequence(wubi_embedding_matrix, ordered_length, batch_first=True)


        char_hidden = None
        rnn_out, char_hidden = self.wubi_lstm(pack_input, char_hidden)
        char_hidden = char_hidden[0].transpose(0, 1).contiguous().view(sentence_length, -1)  # [L * CE], with ordered length
        return char_hidden[idx_back_to_origin_order]

    def wubiEmbed_conv(self, wubi_ids):
        """char_embedding_matrix: [L * LW * CE]
        L: sentence length, LW: word max length, CE: wubi embedding dim
        """
        wubi_embedding_matrix = self.wubi_embedding(wubi_ids)
        wubi_embedding_matrix = wubi_embedding_matrix.transpose(1, 2)  # [L * CE * LW], transpose for conv1d
        wubi_out = self.wubi_cnn(wubi_embedding_matrix)  # [L, out_channels, L]
        wubi_out = nnf.max_pool1d(wubi_out, wubi_out.shape[2])  # [L, out_channels, 1]
        return wubi_out.squeeze(-1)

    def encoder(self, input_ids, wubi_ids, word_lengths):
        embeddings = self.lookup_table(input_ids)
        embeddings = self.embed_drop(embeddings)
        wubi_embedding_list = []
        for one_sent_id, one_sent_word_length in zip(wubi_ids, word_lengths):
            if self.wubi_mode == "CNN":
                wubi_output = self.wubiEmbed_conv(one_sent_id)
            else:
                wubi_output = self.wubiEmbed_rnn(one_sent_id, one_sent_word_length)
            wubi_embedding_list.append(wubi_output)
        wubi_embedding_all = torch.stack(wubi_embedding_list, dim=0)  # [B * L * CE]
        embeddings = torch.cat((embeddings, wubi_embedding_all), dim=-1)
        return embeddings

    def forward(self, input_ids, wubi_ids, true_label_list, masks, word_lengths):
        embeddings = self.encoder(input_ids, wubi_ids, word_lengths)

        whole_auxiliary_loss = 0
        loss_auxiliary, logits_auxiliary = self.Auxiliary_part1.forward(embeddings, true_label_list, masks)
        pred_label_list = self.Auxiliary_part1.predict(logits_auxiliary)  # [B * L]
        label_embeddings = self.label_embedding(pred_label_list)  # [B * L * tag_dim]

        whole_auxiliary_loss += loss_auxiliary

        for i in range(self.auxiliary_num-1):
            whole_embeddings = torch.cat((embeddings, label_embeddings), dim=-1)
            auxiliary = getattr(self, "Auxiliary_part{}".format(i+2))
            loss_auxiliary, logits_auxiliary = auxiliary(whole_embeddings, true_label_list, masks)
            whole_auxiliary_loss += loss_auxiliary
            pred_label_list = auxiliary.predict(logits_auxiliary)  # [B * L]
            label_embeddings = self.label_embedding(pred_label_list)  # [B * L * tag_dim]

        feats = self.Main.get_feats(embeddings, label_embeddings)

        for i in range(self.refactor_num):
            pred_label_list = self.Main.predict(feats)
            pred_label_list = torch.tensor(pred_label_list).to(self.device)
            label_embeddings = self.label_embedding(pred_label_list)
            feats = self.Main.get_feats(embeddings, label_embeddings)

        loss_main = self.Main.forward(feats, true_label_list, masks)
        return loss_main + self.loss_lambda * whole_auxiliary_loss

    def predict(self, input_ids, wubi_ids, masks, sents_word_lengths, idx_to_tag=None):
        embeddings = self.encoder(input_ids, wubi_ids, sents_word_lengths)
        logits = self.Auxiliary_part1.forward(embeddings)

        pred_label_list = self.Auxiliary_part1.predict(logits)  # [B * L]
        label_embeddings = self.label_embedding(pred_label_list)  # [B * L * tag_dim]
        if idx_to_tag is not None:
            pred_label_ids = self.Auxiliary_part1.predict(logits, masks)
            pred_label_tags = [[idx_to_tag[idx] for idx in sent_label_ids] for sent_label_ids in pred_label_ids]

        for i in range(self.auxiliary_num-1):
            whole_embeddings = torch.cat((embeddings, label_embeddings), dim=-1)
            auxiliary = getattr(self, "Auxiliary_part{}".format(i+2))
            logits = auxiliary(whole_embeddings)
            pred_label_list = auxiliary.predict(logits)  # [B * L]
            if idx_to_tag is not None:
                pred_label_ids = auxiliary.predict(logits, masks)
                pred_label_tags = [[idx_to_tag[idx] for idx in sent_label_ids] for sent_label_ids in pred_label_ids]
            label_embeddings = self.label_embedding(pred_label_list)  # [B * L * tag_dim]

        feats = self.Main.get_feats(embeddings, label_embeddings)
        main_pred_label_list = self.Main.predict(feats, masks)
        return main_pred_label_list