#! -*- coding:utf-8 -*-
# global_pointer用来做实体识别
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# 博客：https://kexue.fm/archives/8373
# [valid_f1]: 95.66

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

config = Config()

maxlen = 256
batch_size = 4

# BERT base
config_path = config.config_path
checkpoint_path = config.checkpoint_path
dict_path = config.dict_path
# config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
# dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed
seed_everything(42)


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        data = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):
                if not l:
                    continue
                text, label = '', []
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    text += char
                    if flag[0] == 'B':
                        label.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        label[-1][1] = i
                data.append((text, label))  # label为[[start, end, entity], ...]
        return data


categories_label2id = dict()


def load_data(file_path):
    data = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            line = line.replace("\n", "")
            info = json.loads(line)
            entity_label = [[e[2], e[3], e[0]] for e in info["query_entity"]]
            text = info["query"]
            for e in info["query_entity"]:
                categories_label2id.setdefault(e[0], len(categories_label2id))
            data.append((text, entity_label))
    return data


load_data("../data/Xeon3NLP_round1_train_ner_20210524.txt")

# categories_label2id = {"LOC": 0, "ORG": 1, "PER": 2}
categories_id2label = dict((value, key) for key, value in categories_label2id.items())
ner_vocab_size = len(categories_label2id)
ner_head_size = 64


class LoadMultiTaskData(ListDataset):
    @staticmethod
    def load_data(file_path):
        data = []
        with open(file_path, encoding="utf-8") as file:
            if "dev" in file_path:
                file_data = file.readlines()[-1000:]
            else:
                file_data = file.readlines()[:-1000]
            for line in file_data:
                line = line.replace("\n", "")
                info = json.loads(line)
                entity_label = [[e[2], e[3], e[0]] for e in info["query_entity"]]
                text = info["query"]
                if "dev" not in file_path:
                    for e in info["query_entity"]:
                        categories_label2id.setdefault(e[0], len(categories_label2id))
                data.append((text, entity_label))
        return data


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


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


# 转换数据集
# train_dataloader = DataLoader(MyDataset(config.train_path), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# valid_dataloader = DataLoader(MyDataset(config.dev_path), batch_size=batch_size, collate_fn=collate_fn)
train_dataloader = DataLoader(LoadMultiTaskData(config.train_path), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
valid_dataloader = DataLoader(LoadMultiTaskData(config.train_path), batch_size=batch_size, collate_fn=collate_fn)


class MyLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        return super().forward(y_pred, y_true)


# 定义bert上的模型结构
class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path,
                                            checkpoint_path=checkpoint_path,
                                            segment_vocab_size=0,
                                            model='albert',
                                            )
        self.global_pointer = GlobalPointer(hidden_size=768, heads=ner_vocab_size, head_size=ner_head_size)

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        logit = self.global_pointer(sequence_output, token_ids.gt(0).long())
        return logit


model = Bert()
model.to(device)
loss = MyLoss()

opt = torch.optim.AdamW(model.parameters(), lr=1 * 10e-5)

for epoch in range(10):
    with tqdm(total=int(train_dataloader / batch_size) + 1) as t:
        for batch_token_ids, batch_labels in tqdm(train_dataloader):
            t.set_description('Epoch %i' % epoch)
            batch_token_ids = batch_token_ids.to(device)
            batch_labels = batch_labels.to(device)
            predict = model(batch_token_ids)
            l = loss(predict, batch_labels)
            l.backward()
            opt.step()
            opt.zero_grad()
            t.set_postfix(loss=l)
