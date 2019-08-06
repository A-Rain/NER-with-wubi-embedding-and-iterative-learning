import os
import json
import torch
import logging
import random
import numpy as np
from tqdm import tqdm, trange

from config import hyper_basic, hyper_new_model_wubi_label, finance_data
from model.BCCWL import New_Model_Wubi_label
from utils import *
from data_loader import Char_lstm_Dataloader
from metric import batch_compute_metrics, compute_precision_recall_wrapper
from early_stopping import EarlyStopping

from torch.optim.adam import Adam
from sklearn_crfsuite.metrics import flat_classification_report
import argparse
import wubi


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def val_link(model, dev_loader, device, labels, target_tag_no_schema, idx_to_tag, timeline=0):
    val_losses = []
    true_label_tags, pred_label_tags = [], []
    model.eval()
    with torch.no_grad():
        for num, batch in enumerate(dev_loader.sequential_loading()):
            sents_id, wubi_id, labels_ids, sents_mask, sents_word_lengths = batch.__getitem__()
            sents_id = torch.LongTensor(sents_id).to(device)
            wubi_id = torch.LongTensor(wubi_id).to(device)
            sents_mask = torch.ByteTensor(sents_mask).to(device)
            true_label_tags += [[idx_to_tag[idx] for idx in sent_label_ids] for sent_label_ids in labels_ids]
            max_batch_length = len(sents_id[-1])
            label_ids_padding = padding_batch_of_sequence(labels_ids, max_batch_length, 0)
            label_ids_padding = torch.LongTensor(label_ids_padding).to(device)

            loss = model(sents_id, wubi_id, label_ids_padding, sents_mask, sents_word_lengths)
            pred_label_id_list = model.predict(sents_id, wubi_id, sents_mask, sents_word_lengths, idx_to_tag)
            pred_label_tags += [[idx_to_tag[idx] for idx in sent_label_ids] for sent_label_ids in pred_label_id_list]
            val_losses.append(loss.item())

    logger.info("\n" + flat_classification_report(y_pred=pred_label_tags, y_true=true_label_tags, labels=labels))
    result = batch_compute_metrics(true_label_tags, pred_label_tags, target_tag_no_schema)
    final_result = compute_precision_recall_wrapper(result)
    logger.info("Measure: strict| P: {:.3f}|R: {:.3f}|F1: {:.3f}".format(final_result["strict"]["precision"],
                                                                         final_result["strict"]["recall"],
                                                                         final_result["strict"]["Macro_F1"]))
    return np.average(val_losses)


def train(model, hp, train_loader, dev_loader, optimizer, device, idx_to_tag):
    early_stop_control = EarlyStopping(patience=hp['patience'], verbose=True, ckpt_dir=hp['save_dir'])
    whole_step = 0
    avg_train_losses, avg_val_losses = [], []
    train_losses, valid_losses = [], []

    validation_step = int(train_loader.steps()*0.3)
    model.train()
    for epoch in range(hp['epoch']):
        if early_stop_control.early_stop:
            logger.info("meet early stopping, jump out")
            break
        for No, batch in enumerate(train_loader.Random_loading()):
            whole_step += 1
            sents_id, wubi_id, labels_ids, sents_mask, sents_word_lengths = batch.__getitem__()
            sents_id = torch.LongTensor(sents_id).to(device)
            wubi_id = torch.LongTensor(wubi_id).to(device)
            sents_mask = torch.ByteTensor(sents_mask).to(device)
            max_batch_length = len(sents_id[-1])
            label_ids_padding = padding_batch_of_sequence(labels_ids, max_batch_length, 0)
            # print("labels: ", label_ids_padding)
            label_ids_padding = torch.LongTensor(label_ids_padding).to(device)

            optimizer.zero_grad()
            loss = model(sents_id, wubi_id, label_ids_padding, sents_mask, sents_word_lengths)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if whole_step%100==0:
                logger.info("epoch {} | batch step {} | batch loss: {:.3f}".format(epoch, No, loss.item()))

            if whole_step % validation_step == 0:
                val_loss = val_link(model, dev_loader, device, hp['labels'], hp['entity_type'], idx_to_tag, whole_step)
                early_stop_control(val_loss, model)
                logger.info("avg training loss: {}, avg validation loss: {}".format(np.average(train_losses), val_loss))

                avg_val_losses.append(val_loss)


                model.train()

            if early_stop_control.early_stop:
                logger.info("meet early stopping, jump out")
                break

        avg_train_losses.append(np.average(train_losses))
        train_losses = []


    with open(hp["loss_record_dir"], mode='w', encoding="utf-8") as f:
        f.writelines(str(avg_train_losses)+'\n')
        f.writelines(str(avg_val_losses))

    model.load_state_dict(torch.load(hp['save_dir']))
    return model


def test(model, test_loader, device, target_tag_no_schema, eval_record_dir, idx_to_tag, labels):
    model.eval()
    with torch.no_grad():
        true_label_tags, pred_label_tags = [], []
        for batch in tqdm(test_loader.sequential_loading(), desc="test Iteration", total=test_loader.steps()):
            sents_id, wubi_id, labels_ids, sents_mask, sents_word_lengths = batch.__getitem__()
            sents_id = torch.LongTensor(sents_id).to(device)
            wubi_id = torch.LongTensor(wubi_id).to(device)
            sents_mask = torch.ByteTensor(sents_mask).to(device)
            true_label_tags += [[idx_to_tag[idx] for idx in sent_label_ids] for sent_label_ids in labels_ids]

            pred_seq_labes_ids = model.predict(sents_id, wubi_id, sents_mask, sents_word_lengths)
            pred_label_tags += [[idx_to_tag[idx] for idx in sent_label_ids] for sent_label_ids in pred_seq_labes_ids]

        result = batch_compute_metrics(true_label_tags, pred_label_tags, target_tag_no_schema)
        final_result = compute_precision_recall_wrapper(result)
        logger.info("\n" + flat_classification_report(y_pred=pred_label_tags, y_true=true_label_tags, labels=labels))
        logger.info("Measure: ent_type| P: {:.4f}|R: {:.4f}|F1: {:.4f}".format(final_result["ent_type"]["precision"],
                                                                           final_result["ent_type"]["recall"],
                                                                           final_result["ent_type"]["Macro_F1"]))
        logger.info("Measure: partial| P: {:.4f}|R: {:.4f}|F1: {:.4f}".format(final_result["partial"]["precision"],
                                                                           final_result["partial"]["recall"],
                                                                           final_result["partial"]["Macro_F1"]))
        logger.info("Measure: strict| P: {:.4f}|R: {:.4f}|F1: {:.4f}".format(final_result["strict"]["precision"],
                                                                           final_result["strict"]["recall"],
                                                                           final_result["strict"]["Macro_F1"]))
        logger.info("Measure: exact| P: {:.4f}|R: {:.4f}|F1: {:.4f}".format(final_result["exact"]["precision"],
                                                                           final_result["exact"]["recall"],
                                                                           final_result["exact"]["Macro_F1"]))
        with open(eval_record_dir, mode="w", encoding="utf-8") as eval_file:
            json.dump(dict(result), eval_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        help="choose dataset, could be msra, finance, weibo or people_daily")
    parser.add_argument('--wubi_mode',
                        type=str,
                        default='CNN',
                        help="choose wubi embedding mode, could be CNN/RNN")
    parser.add_argument('--aux_num',
                        type=int,
                        default=2,
                        help="choose auxiliary num, could be 1/2/3/4")
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help="choose dropout rate")
    parser.add_argument('--wubi_freeze',
                        type=int,
                        default=1)
    parser.add_argument('--char_freeze',
                        type=int,
                        default=1)

    args = parser.parse_args()

    data = {
        "finance": finance_data,
    }

    tag_to_idx = {
        "finance": tag_to_idx_finance,
    }

    hp = hyper_basic
    hp.update(data[args.dataset])
    hp.update(hyper_new_model_wubi_label)

    labels = list(tag_to_idx[args.dataset].keys())
    labels.remove('O')
    hp.update({'labels': labels})

    hp['auxiliary_num'] = args.aux_num
    hp['auxiliary_config']['dropout'] = args.dropout
    hp['main_config']['dropout'] = args.dropout

    predix = "{}_{}_{}_{}_{}_".format(args.char_freeze, args.wubi_freeze, args.wubi_mode, args.aux_num, args.dropout)
    hp["save_dir"] = os.path.join("./model_save", args.dataset, predix+hp["save_dir"])
    hp["loss_record_dir"] = os.path.join("./record", args.dataset, predix+hp["loss_record_dir"])
    hp["eval_record_dir"] = os.path.join("./record", args.dataset, predix+hp["eval_record_dir"])

    if os.path.exists(os.path.join(hp['cache_dir'], "weights.npy")) \
        and os.path.exists(os.path.join(hp['cache_dir'], "vocab.json")):
        vocab = json.load(open(os.path.join(hp['cache_dir'], "vocab.json")))
        weights = np.load(os.path.join(hp['cache_dir'], "weights.npy"))
    else:
        _, vocab_train = data_reader(hp['dataset_train'], data_type=args.dataset, length_cut_off=hp['len_cut_off'])
        _, vocab_test = data_reader(hp['dataset_test'], data_type=args.dataset, length_cut_off=hp['len_cut_off'])
        if args.dataset in ["people_daily", "weibo"]:
            _, vocab_dev = data_reader(hp['dataset_dev'], data_type=args.dataset, length_cut_off=hp['len_cut_off'])
            vocab = vocab_train | vocab_test | vocab_dev
        else:
            vocab = vocab_train | vocab_test
        weights, vocab = get_embedding_weights_and_vocab(vocab, hp['pretrain_embedding'], hp['embedding_dim'])
        np.save(os.path.join(hp['cache_dir'], "weights"), weights)
        json.dump(vocab, open(os.path.join(hp['cache_dir'], "vocab.json"), mode="w", encoding="utf-8"), ensure_ascii=False)
    with open(hp["char_vocab"], mode='r', encoding="utf-8") as f:
        char_vocab = json.load(f)
    logger.info("finishing loading vocabulary and embedding weights ...")

    ### read corpus
    corpus_train, _ = data_reader(hp['dataset_train'], data_type=args.dataset, length_cut_off=hp['len_cut_off'])
    corpus_test, _ = data_reader(hp['dataset_test'], data_type=args.dataset, length_cut_off=hp['len_cut_off'])
    if args.dataset in ["people_daily", "weibo"]:
        corpus_dev, _ = data_reader(hp['dataset_dev'], data_type=args.dataset, length_cut_off=hp['len_cut_off'])
    else:
        corpus_train, corpus_dev = train_test_split(corpus_train, hp['dev_size'])

    ### build training dataloader
    wubi_tokenizer = wubi.data['cw']
    train_loader = Char_lstm_Dataloader(corpus_train, hp['batch_train'])
    train_loader.create_one_batch(vocab, tag_to_idx[args.dataset], hp['wubi_max_length'], char_vocab, wubi_tokenizer, args.wubi_mode)

    ### build dev dataloader
    dev_loader = Char_lstm_Dataloader(corpus_dev, hp['batch_train'])
    dev_loader.create_one_batch(vocab, tag_to_idx[args.dataset], hp['wubi_max_length'], char_vocab, wubi_tokenizer, args.wubi_mode)

    ### build test dataloader
    test_loader = Char_lstm_Dataloader(corpus_test, hp['batch_eval'])
    test_loader.create_one_batch(vocab, tag_to_idx[args.dataset], hp['wubi_max_length'], char_vocab, wubi_tokenizer, args.wubi_mode)

    logger.info("finishing build train/test/dev data_loader ...")
    logger.info("*************Data Loader INFO*****************")
    logger.info("batch size for trainig/dev: {}".format(hp['batch_train']))
    logger.info("step for train: {}, step for dev: {}".format(train_loader.steps(), dev_loader.steps()))
    logger.info("**********************************************")
    logger.info("batch size for test: {}".format(hp['batch_eval']))
    logger.info("step for test: {}".format(test_loader.steps()))
    logger.info("**********************************************")

    ### loading model
    device = torch.device("cuda" if torch.cuda.is_available and hp['use_gpu'] else "cpu")
    hp['auxiliary_config']["device"] = device
    hp['main_config']["device"] = device
    weights = torch.from_numpy(weights).float()
    model = New_Model_Wubi_label(
        num_labels=len(tag_to_idx[args.dataset]),
        alphabet_size=len(char_vocab),
        wubi_embedding_dim=hp['wubi_embed'],
        wubi_out_dim=hp['wubi_embed'],
        embedding_weights=weights,
        refactor_num=hp['refactor_num'],
        auxiliary_num=hp['auxiliary_num'],
        auxiliary_config=hp['auxiliary_config'],
        main_config=hp['main_config'],
        label_embedding_dim=hp['label_embed'],
        loss_lambda=hp['lambda'],
        freeze_embed=args.char_freeze,
        wubi_mode=args.wubi_mode,
        wubi_freeze=args.wubi_freeze
    )

    model.to(device)
    optimizer = Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

    logger.info("begin training ...")
    idx_to_tag = dict([(i[1], i[0]) for i in tag_to_idx[args.dataset].items()])
    best_model = train(model, hp, train_loader, dev_loader, optimizer, device, idx_to_tag)
    test(best_model, test_loader, device, hp['entity_type'], hp['eval_record_dir'], idx_to_tag, labels)


if __name__ == "__main__":
    main()

