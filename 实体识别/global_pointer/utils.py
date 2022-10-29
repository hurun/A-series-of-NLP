import numpy as np
from bert4torch.models import build_transformer_model, BaseModel
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.tokenizers import Tokenizer
from bert4torch.losses import MultilabelCategoricalCrossentropy
from bert4torch.layers import GlobalPointer
import random
import os
import json
from config import Config
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertTokenizerFast

config = Config()

tokenizer = BertTokenizerFast.from_pretrained(config.bert_dir, add_special_tokens=True, do_lower_case=False)
dict_path = config.dict_path

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(file_path):
        data = []
        with open(file_path, encoding="utf-8") as file:
            file_data = file.readlines()
            for line in file_data:
                line = line.replace("\n", "")
                info = json.loads(line)
                entity_label = [[e[2], e[3], e[0]] for e in info["query_entity"]]
                text = info["query"]
                data.append((text, entity_label))
        return data


def get_categories_label2id(file_path):
    categories_label2id = dict()
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            line = line.replace("\n", "")
            info = json.loads(line)

            for e in info["query_entity"]:
                categories_label2id.setdefault(e[0], len(categories_label2id))
    return categories_label2id


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
categories_label2id = get_categories_label2id(config.train_path)
maxlen = config.maxlen
device = config.device

def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for i, (text, text_labels) in enumerate(batch):
        tokens = tokenizer.tokenize(text, maxlen=maxlen)
        mapping = tokenizer.rematch(text, tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros((len(categories_label2id), maxlen, maxlen))
        for start, end, label in text_labels:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                label = categories_label2id[label]
                labels[label, start, end] = 1

        batch_token_ids.append(token_ids)  # 前面已经限制了长度
        batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels, seq_dims=3), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels


class MyLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        return super().forward(y_pred, y_true)