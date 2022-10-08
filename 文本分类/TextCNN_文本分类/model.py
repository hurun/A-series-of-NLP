import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, kernel_s, embeddin_num, max_len, hidden_num):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(
            kernel_s, embeddin_num))  # 1 * 1 * 7 * 5 (batch *  in_channel * len * emb_num )
        self.act = nn.ReLU()
        self.mxp = nn.MaxPool1d(kernel_size=(max_len - kernel_s + 1))

    def forward(self, batch_emb):  # 1 * 1 * 7 * 5 (batch *  in_channel * len * emb_num )
        c = self.cnn.forward(batch_emb)
        a = self.act.forward(c)
        a = a.squeeze(dim=-1)
        m = self.mxp.forward(a)
        m = m.squeeze(dim=-1)
        return m


class TextCNNModel(nn.Module):
    def __init__(self, emb_matrix, max_len, class_num, hidden_num):
        super().__init__()
        self.emb_num = emb_matrix.weight.shape[1]

        self.block1 = Block(2, self.emb_num, max_len, hidden_num)
        self.block2 = Block(3, self.emb_num, max_len, hidden_num)
        self.block3 = Block(4, self.emb_num, max_len, hidden_num)
        self.block4 = Block(5, self.emb_num, max_len, hidden_num)

        self.emb_matrix = emb_matrix

        self.classifier = nn.Linear(hidden_num * 4, class_num)  # 2 * 3
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_idx, batch_label=None):
        batch_emb = self.emb_matrix(batch_idx)
        b1_result = self.block1.forward(batch_emb)
        b2_result = self.block2.forward(batch_emb)
        b3_result = self.block3.forward(batch_emb)
        b4_result = self.block4.forward(batch_emb)

        feature = torch.cat([b1_result, b2_result, b3_result, b4_result], dim=1)  # 1* 6 : [ batch * (3 * 2)]
        pre = self.classifier(feature)
        return pre
