import unicodedata, re
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import BertTokenizerFast, BertTokenizer


class token_rematch:
    def __init__(self):
        self._do_lower_case = True

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end
        return token_mapping


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        """ y_true ([Tensor]): [..., num_classes]
            y_pred ([Tensor]): [..., num_classes]
        """
        y_pred = (1-2*y_true) * y_pred
        y_pred_pos = y_pred - (1-y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()

class MyLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        return super().forward(y_pred, y_true)

class MyDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length

    def load_data(self, file_path):
        data = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                item = {}
                item["text"] = line["text"]
                item["entity_list"] = []
                for k, v in line['label'].items():
                    for spans in v.values():
                        for start, end in spans:
                            item["entity_list"].append((start, end, k))
                data.append(item)
        return data


def collate_fn(batch_data, tokenizer: BertTokenizerFast, entity_to_id: dict, max_len: int):
    batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_labels = [], [], [], []
    for idx, info in enumerate(batch_data):
        text = info["text"]
        inputs = tokenizer(text, return_offsets_mapping=True,
                           max_length=max_len,
                           padding="max_length",
                           truncation=True,
                           add_special_tokens=True,
                           return_tensors="pt")
        label = torch.zeros(len(entity_to_id), max_len, max_len)
        tokens = tokenizer.tokenize(text,add_special_tokens=True)
        mapping = token_rematch().rematch(text, tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}

        # entity index rematch label
        for entity in info["entity_list"]:
            start, end = entity[0], entity[1]
            entity_type = entity_to_id[entity[2]]
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                label[entity_type, start, end] = 1
        batch_input_ids.append(torch.tensor(inputs["input_ids"][0]).long())
        batch_attention_mask.append(torch.tensor(inputs["attention_mask"][0]).long())
        batch_token_type_ids.append(torch.tensor(inputs["token_type_ids"][0]).long())
        batch_labels.append(torch.tensor(label).long())

    # single data to batch data
    batch_input_ids = torch.stack(batch_input_ids, dim=0)
    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    batch_token_type_ids = torch.stack(batch_token_type_ids, dim=0)
    batch_labels = torch.stack(batch_labels, dim=0)

    return batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels


if __name__ == "__main__":
    mydata = MyDataset("./data/cluener/train.json")
    print(mydata.length)
    print(mydata.__getitem__(2))
