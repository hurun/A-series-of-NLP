from typing import Optional, List
import json
import random

file_path = "./Xeon3NLP_round1_train_ner_20210524.txt"

data = []
with open(file_path, encoding="utf-8") as file:
    for line in file:
        data.append(line)

random.shuffle(data)


def write_file_to_txt(save_file_path, data_list: Optional[List[str]]):
    with open(save_file_path, "w", encoding="utf-8") as file:
        for line in data_list:
            file.writelines(line)


write_file_to_txt("./train_xeon3nlp.txt", data[:int(len(data) * 0.9)])
write_file_to_txt("./valid_xeon3nlp.txt", data[int(len(data) * 0.9): int(len(data) * 0.95)])
write_file_to_txt("./test_xeon3nlp.txt", data[:int(len(data) * 0.95):])
