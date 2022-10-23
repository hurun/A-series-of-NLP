import torch

class Config():
    def __init__(self):
        self.director_path = "../data/"
        self.train_path = self.director_path + "train_xeon3nlp.txt"
        self.dev_path = self.director_path + "valid_xeon3nlp.txt"
        self.test_path = self.director_path + "test_xeon3nlp.txt"
        self.all_schema = self.director_path + "all_schemas"

        self.maxlen = 64
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bert
        self.model_name = "bert"
        self.bert_dir = "/home/hurun/BertModel/torch/albert-base-v2/"
        self.config_path = self.bert_dir + 'config.json'
        self.checkpoint_path = self.bert_dir + 'pytorch_model.bin'
        self.dict_path = self.bert_dir + 'vocab_chinese.txt'
        self.onnx_model_path = "./model/bert.onnx"
        self.optimized_onnx_model_path = "./model/bert.onnx"
        self.model_path = "/home/guest/PycharmProjects/GPLinker/bert_best_model.weights"

        self.model_path = "./model/best_model_cmeee_globalpointer_0805_pp.weights"
