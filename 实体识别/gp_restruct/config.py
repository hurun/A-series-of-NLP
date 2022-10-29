import torch

class Config():
    def __init__(self):
        self.director_path = "./data"
        self.train_path = self.director_path + "/cluener/train.json"
        self.dev_path = self.director_path + "/cluener/dev.json"
        self.test_path = self.director_path + "/cluener/test.json"
        self.entity_to_id = self.director_path + "/cluener/ent2id.json"

        self.max_len = 64
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 32
        # bert
        self.model_name = "bert"
        self.bert_dir = "/home/hurun/BertModel/torch/bert-base-chinese/"
        self.config_path = self.bert_dir + 'config.json'
        self.bert_model = self.bert_dir + 'pytorch_model.bin'
        self.dict_path = self.bert_dir + 'vocab_chinese.txt'
        self.onnx_model_path = "./model/bert.onnx"
        self.optimized_onnx_model_path = "./model/bert.onnx"
        self.model_path = "/home/guest/PycharmProjects/GPLinker/bert_best_model.weights"

        self.model_path = "./model/best_model_cmeee_globalpointer_0805_pp.weights"
