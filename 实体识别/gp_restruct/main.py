from transformers import BertTokenizerFast, BertModel
import torch
from tqdm import tqdm
from util import *
from model import *
from config import Config

conf = Config()

# text to token
tokenizer = BertTokenizerFast.from_pretrained(conf.bert_dir, add_special_tokens=True, do_lower_case=False)

# load data
train_data = MyDataset(conf.train_path)
dev_data = MyDataset(conf.dev_path)
# test_data = MyDataset(conf.test_path)

entity_to_id = json.load(open(conf.entity_to_id), encoding="utf-8")
id_to_entity = {v: k for k, v in entity_to_id.items()}
entity_size = len(entity_to_id)

# dataset to data_loader

train_data_loader = DataLoader(train_data, batch_size=conf.batch_size, num_workers=2,
                               collate_fn=lambda x: collate_fn(x, tokenizer, entity_to_id, conf.max_len))
dev_data_loader = DataLoader(dev_data, batch_size=conf.batch_size, num_workers=2,
                             collate_fn=lambda x: collate_fn(x, tokenizer, entity_to_id, conf.max_len)
                             )
# test_data_loader = DataLoader(test_data, batch_size=2, num_workers=2)

model = NerModel(conf.bert_dir, entity_size, 64)
model = model.to(conf.device)
loss_func = MyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

if __name__ == "__main__":
    best_f1 = 0.0
    threshold = 0
    for epoch in range(10):
        model.train()
        with tqdm(total=int(len(train_data) / conf.batch_size)) as t:
            for batch_data in train_data_loader:
                batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
                batch_input_ids = batch_input_ids.to(conf.device)
                batch_attention_mask = batch_attention_mask.to(conf.device)
                batch_token_type_ids = batch_token_type_ids.to(conf.device)
                batch_labels = batch_labels.to(conf.device)

                logit = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                loss = loss_func(logit, batch_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                t.set_description('Epoch %i' % epoch)
                t.set_postfix(loss=loss.item())
                t.update(1)

        model.eval()
        with tqdm(total=int(len(dev_data) / conf.batch_size)) as t:
            with torch.no_grad():
                X, Y, Z = 0, 1e-10, 1e-10
                for batch_data in dev_data_loader:
                    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
                    batch_input_ids = batch_input_ids.to(conf.device)
                    batch_attention_mask = batch_attention_mask.to(conf.device)
                    batch_token_type_ids = batch_token_type_ids.to(conf.device)
                    batch_labels = batch_labels.to(conf.device)

                    predict = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                    for i, score in enumerate(predict):
                        R = set()
                        for l, start, end in zip(*np.where(score.cpu() > threshold)):
                            R.add((start, end, id_to_entity[l]))

                        T = set()
                        for l, start, end in zip(*np.where(batch_labels[i].cpu() > threshold)):
                            T.add((start, end, id_to_entity[l]))
                        X += len(R & T)
                        Y += len(R)
                        Z += len(T)
                    t.set_description('Epoch %i' % epoch)
                    t.set_postfix({"f1": "{:.4f}".format(2 * X / (Y + Z)),
                                   "precision": "{:.4f}".format(X / Y),
                                   "recall": "{:.4f}".format(X / Z)
                                   })
                    t.update(1)
                f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
                print(f1, precision, recall)
