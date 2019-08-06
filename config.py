finance_data = {
    "dataset_train"     :   "./data/finance/train_finance_BIO.txt",
    "dataset_test"      :   "./data/finance/test_finance_BIO.txt",
    "cache_dir"         :   "./cache/finance",
    "entity_type"       :   ['PER', 'CPY', 'BRAND', 'MISC']
}


hyper_basic = {
    "pretrain_embedding":   "./pretrain_embedding/sgns.renmin.bigram-char",
    "batch_train"       :   16,
    "batch_eval"        :   8,
    "embedding_dim"     :   300,
    "dev_size"          :   0.1,
    "lr"                :   1e-4,
    "weight_decay"      :   0.01,
    "epoch"             :   10,
    "patience"          :   7,
    "len_cut_off"       :   150,
    "use_gpu"           :   True,
}


hyper_new_model_wubi_label = {
    "save_dir"          :   "checkpoint_new_model_wubi_half_freeze.pt",
    "loss_record_dir"   :   "loss_new_model_wubi_half_freeze.txt",
    "eval_record_dir"   :   "eval_new_model_wubi_half_freeze.json",
    "char_vocab"        :   "./cache/char_vocab.json",
    "refactor_num"      :   0,
    "auxiliary_num"     :   3,
    "freeze_embed"      :   False,
    "lambda"            :   10,
    "label_embed"       :   30,
    "wubi_embed"        :   24,
    "wubi_max_length"   :   5,
    "auxiliary_config"  :   {
        "input_size"    :   300,
        "hidden_size"   :   200,
        "bidirectional" :   True,
        "batch_first"   :   True,
        "num_layers"    :   2,
        "RNNcell"       :   "LSTM",
        "IsDropout"     :   True,
        "dropout"       :   0.4,
        "IsDropConnect" :   False,
        "weightdrop"    :   0.4
    },
    "main_config"       :   {
        "input_size"    :   300,
        "hidden_size"   :   200,
        "bidirectional" :   True,
        "batch_first"   :   True,
        "num_layers"    :   2,
        "RNNcell"       :   "LSTM",
        "IsDropout"     :   True,
        "dropout"       :   0.4,
        "IsDropConnect" :   False,
        "weightdrop"    :   0.4
    }
}