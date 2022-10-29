from bert4torch.layers import GlobalPointer
import torch.nn as nn
from transformers import BertModel

class NerModel(nn.Module):
    def __init__(self, bert_model_path: str, entity_type_nums: int, head_size: int):
        super(NerModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_path)
        self.global_pointer = GlobalPointer(hidden_size=768, heads=entity_type_nums, head_size=head_size)

    def forward(self, batch_input_ids, batch_attention_mask, batch_token_type_ids):
        sequence_outputs = self.bert_model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        logit = self.global_pointer(sequence_outputs[0], batch_input_ids.gt(0).long())
        return logit