import os
import json
from pipetools import *
import pipetools
from torch.utils.data import Dataset, DataLoader
from bert4torch.tokenizers import Tokenizer
import torch
import numpy as np
from bert4torch.snippets import sequence_padding

torch_bert_base_dir = "/home/hurun/BertModel/bert-base-chinese/"
torch_bert_base_dict_path = torch_bert_base_dir + 'vocab.txt'

tokenizer = Tokenizer(torch_bert_base_dict_path, do_lower_case=True)
label = {"不匹配": 0, "部分匹配": 1, "完全匹配": 2}

category2id = {'town': 0, 'prov': 1, 'houseno': 2, 'poi': 3, 'city': 4, 'road': 5, 'cellno': 6, 'roadno': 7,
               'floorno': 8, 'district': 9, 'devzone': 10, 'community': 11, 'subpoi': 12, 'distance': 13,
               'village_group': 14, 'intersection': 15}


def read_xeon3(file_path):
    _data_ = []
    query = set()
    label = dict()
    with open(file_path) as file:
        for line in file:
            line = line.replace("\n", "")
            info = json.loads(line)
            # query.add(info["query"])
            _data_.append(info)
            # for qe in info["query_entity"]:
            #     category2id.setdefault(qe[0], len(category2id))

            for candidate in info['candidate']:
                _data_.append({"source_text": info["query"],
                               "target_text": candidate["text"],
                               "similar": label.get(candidate["label"], len(label)),
                               "source_entity": info["query_entity"],
                               "target_entity": candidate["text_entity"]
                               })
    print(category2id)
    return query, _data_


def read_conll(file_path):
    _data_ = {}
    con_query = dict()
    label = set()
    max_len = 0
    with open(file_path) as file:
        tmp = []
        query = ""
        for line in file:
            # line = line.replace("n")
            if line == "\n":
                max_len = max(max_len, len(query))
                con_query[query] = tmp
                tmp = []
                query = ""
            else:
                line = line.replace("\n", "")
                word, ner_lab = line.split(" ")
                query += word
                label.add(ner_lab)
                tmp.append(ner_lab)
    print(max_len)
    return con_query


def extract_same_data(xeon3, conll):
    xeon3_query, xeon3_data = read_xeon3("./data/Xeon3NLP_round1_train_20210524.txt")
    conll_data = read_conll("./data/train.conll")
    print(len(xeon3_query & set(conll_data.keys())))
    pass


class MultiTaskDataset(Dataset):
    def __init__(self, data, max_len=128):
        self.data = data
        self.max_len = max_len

    def __getitem__(self, index):
        source_text = self.data[index]["source_text"]
        target_text = self.data[index]["target_text"]
        similar = self.data[index]["similar"]
        source_entity = self.data[index]["source_entity"]
        target_entity = self.data[index]["target_entity"]
        return source_text, target_text, source_entity, target_entity, int(similar)

    def collate_fn(self, batch_data):
        batch_source_token_ids, batch_target_token_ids = [], []
        batch_source_entity_labels, batch_target_entity_labels = [], []
        batch_similar_labels = []

        for i, (source_text, target_text, source_entity, target_entity, similar) in enumerate(batch_data):
            # target encode & label batch
            target_tokens = tokenizer.tokenize(target_text, maxlen=self.max_len)
            target_tokens_ids = tokenizer.tokens_to_ids(target_tokens)
            batch_target_token_ids.append(target_tokens_ids)
            batch_similar_labels.append(similar)

            # source encode & source entity encode
            source_tokens = tokenizer.tokenize(source_text, maxlen=self.max_len)
            mapping = tokenizer.rematch(source_text, source_tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            source_token_ids = tokenizer.tokens_to_ids(source_tokens)
            labels = np.zeros((len(category2id), self.max_len, self.max_len))
            for cls, ner_text, start, end in source_entity:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = category2id[cls]
                    labels[label, start, end] = 1

            batch_source_token_ids.append(source_token_ids)  # 前面已经限制了长度
            batch_source_entity_labels.append(labels[:, :len(source_token_ids), :len(source_token_ids)])

        # to tensor
        batch_source_token_ids = torch.tensor(sequence_padding(batch_source_token_ids), dtype=torch.long, device="cuda")
        batch_source_entity_labels = torch.tensor(sequence_padding(batch_source_entity_labels, seq_dims=3),
                                                  dtype=torch.long, device="cuda")
        batch_target_token_ids = torch.tensor(sequence_padding(batch_target_token_ids), dtype=torch.long, device="cuda")
        batch_similar_labels = torch.tensor(batch_similar_labels)

        return batch_source_token_ids, batch_target_token_ids, \
               batch_target_entity_labels, batch_source_entity_labels, \
               batch_similar_labels


if __name__ == "__main__":
    # extract_same_data(1, 1)
    read_xeon3("./data/Xeon3NLP_round1_train_ner_20210524.txt")
