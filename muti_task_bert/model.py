import torch
import torch.nn as nn
from transformers import BertModel


class MultiTaskBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("/home/hurun/BertModel/bert-base-chinese/")
        self.classifier = nn.Linear(768, class_num)
        self.loss_fun = nn.CrossEntropyLoss()

        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        pass

    def forward(self, batch_source_token_ids, batch_target_token_ids,
                batch_target_entity_labels, batch_source_entity_labels,
                batch_similar_labels):

        source_out = self.bert(batch_source_token_ids, attention_mask=(batch_source_token_ids > 0))
        target_out = self.bert(batch_target_token_ids, attention_mask=(batch_target_token_ids > 0))

