#! -*- coding:utf-8 -*-
# global_pointer用来做实体识别
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# 博客：https://kexue.fm/archives/8373
# [valid_f1]: 95.66
import torch

from utils import *
from model import *

config = Config()

maxlen = config.maxlen
batch_size = 4

# BERT base
config_path = config.config_path
checkpoint_path = config.checkpoint_path
dict_path = config.dict_path
device = config.device

# 固定seed
seed_everything(42)

train_dataset = MyDataset(config.train_path)
valid_dataset = MyDataset(config.dev_path)
test_dataset = MyDataset(config.test_path)

categories_id2label = dict((value, key) for key, value in categories_label2id.items())
ner_vocab_size = len(categories_label2id)
ner_head_size = 64

# 转换数据集
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

model = Bert(config_path, checkpoint_path, ner_vocab_size, ner_head_size)
model.to(device)
loss = MyLoss()

opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(10):
    model.train()
    with tqdm(total=int(train_dataset.__len__() / batch_size) + 1) as t:
        for batch_token_ids, batch_labels in tqdm(train_data_loader):
            batch_token_ids = batch_token_ids.to(device)
            batch_labels = batch_labels.to(device)
            predict = model(batch_token_ids)
            l = loss(predict, batch_labels)
            l.backward()
            opt.step()
            opt.zero_grad()

            t.set_description('Epoch %i' % epoch)
            t.set_postfix(loss=l.item())
            t.update(1)

    model.eval()
    threshold = 0
    with torch.no_grad():
        X, Y, Z = 0, 1e-10, 1e-10

        for batch_token_ids, batch_labels in tqdm(valid_data_loader):
            batch_token_ids = batch_token_ids.to(device)
            batch_labels = batch_labels.to(device)
            predict = model(batch_token_ids)
            for i, score in enumerate(predict):
                R = set()
                for l, start, end in zip(*np.where(score.cpu() > threshold)):
                    R.add((start, end, categories_id2label[l]))

                T = set()
                for l, start, end in zip(*np.where(batch_labels[i].cpu() > threshold)):
                    T.add((start, end, categories_id2label[l]))
                X += len(R & T)
                Y += len(R)
                Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        print("F1: {}, Precision:{}, Recall:{}".format(f1, precision, recall))
