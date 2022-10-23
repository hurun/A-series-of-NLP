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



# 定义bert上的模型结构
class Bert(nn.Module):
    def __init__(self, config_path, checkpoint_path, ner_vocab_size, ner_head_size):
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

