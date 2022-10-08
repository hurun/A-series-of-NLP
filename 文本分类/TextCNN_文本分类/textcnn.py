from tqdm import tqdm
from torch.utils.data import DataLoader

from model import *
from utils import *


class Trainer(object):
    def __init__(self, emb_dim: int, _batch_size_: int, _epochs_: int,
                 lr: float, out_channel: int, class_num: int,
                 device: str,
                 ):
        self.emb_dim = emb_dim
        self.batch_size = _batch_size_
        self.epochs = _epochs_
        self.lr = lr
        self.out_channel = out_channel
        self.class_num = class_num
        self.device = device
        self.loss = nn.CrossEntropyLoss()
        # self.model = text_cnn_model.to(device)
        # self.opt = optim.AdamW(self.model.parameters(), lr=lr)

    def get_data(self, train_loader, dev_loader):

        self.train_data_loader = train_loader
        self.dev_data_loader = dev_loader

    def train(self):
        model = TextCNNModel(words_embedding, max_len, class_num, conv_out_channel).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fun = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            for batch_idx, batch_label in tqdm(self.train_data_loader):
                batch_idx = batch_idx.to(self.device)
                batch_label = batch_label.to(self.device)
                predict = model(batch_idx, batch_label)
                loss = loss_fun(predict, batch_label)
                loss.backward()
                opt.step()
                opt.zero_grad()
            print(f"loss:{loss:.3f}")

            total, right = 0., 0.
            for batch_idx, batch_label in tqdm(self.dev_data_loader):
                batch_idx = batch_idx.to(self.device)
                batch_label = batch_label.to(self.device)
                predict = model(batch_idx).argmax(dim=-1)
                total += len(predict)
                right += int(torch.sum(predict == batch_label))

            print(right / total, total)
            # self.eval()
        pass

    def eval(self):
        # self.model.eval()
        total, right = 0., 0.
        for batch_idx, batch_label in tqdm(self.dev_data_loader):
            batch_idx = batch_idx.to(self.device)
            batch_label = batch_label.to(self.device)
            predict = self.model(batch_idx).argmax(axis=-1)
            total += len(predict)
            right += int(torch.sum(predict == batch_label))

        print(right / total, total)


if __name__ == "__main__":
    train_text, train_label = read_data("train")
    dev_text, dev_label = read_data("dev")

    embedding = 50
    max_len = 20
    batch_size = 200
    epochs = 100
    lr = 0.001
    conv_out_channel = 2
    class_num = len(set(train_label))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    word_2_index, words_embedding = built_curpus(train_text, embedding)

    train_dataset = TextDataset(train_text, train_label, word_2_index, max_len)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)

    dev_dataset = TextDataset(dev_text, dev_label, word_2_index, max_len)
    dev_loader = DataLoader(dev_dataset, batch_size, shuffle=False)

    trainer = Trainer(embedding, batch_size, epochs, lr, conv_out_channel,
                      class_num, device, )
    trainer.get_data(train_loader, dev_loader)
    trainer.train()
