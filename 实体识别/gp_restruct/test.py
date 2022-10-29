# 与前一个任务重复部分如Trainer，不列出，请见官网

# token级别的分类，对句子中每个token做分类
# 我们这里用最常见的ner作为例子

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

wnut = load_dataset("wnut_17", data_files="./data/wnut17train.conll")  # 用这个数据集得科学上网

# NER的数据集是这样
# wnut["train"][0]
# {'id': '0',
#  'ner_tags': [0, 0, 0, 0, 0, 0, 0,
#               0, 0, 0, 0, 0, 0, 0, 7,
#               8, 8, 0, 7, 0, 0, 0,
#               0, 0, 0, 0, 0],
#  'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where',
#             'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire',
#             'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad',
#             'storm', 'here', 'last', 'evening', '.']
# }

label_list = wnut["train"].features[f"ner_tags"].feature.names

# 每一种ner_tags表明了一种实体，描述实体，比如是机构，地点，或是人物
# 前缀 B 是表明实体的开始
# 前缀 I 表明这个token是实体的一部分
#     0 代表不指向任何一个实体
# 所以上面的例子就是 7代表地点实体的开始，8代表这词也是实体的一部分
# label_list
# [
#     "O",
#     "B-corporation",
#     "I-corporation",
#     "B-creative-work",
#     "I-creative-work",
#     "B-group",
#     "I-group",
#     "B-location",
#     "I-location",
#     "B-person",
#     "I-person",
#     "B-product",
#     "I-product",
# ]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 直接对ner数据集用分词器的话，因为要加上[CLS][SEP]还有会##分词，会破坏标签和token的对应关系
# tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
# tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
# tokens
# ['[CLS]', '@', 'paul', '##walk', 'it', "'", 's', 'the', 'view', 'from', 'where', 'i', "'", 'm', 'living',
# 'for', 'two', 'weeks', '.', 'empire', 'state', 'building', '=', 'es', '##b', '.', 'pretty', 'bad',
# 'storm', 'here', 'last', 'evening', '.', '[SEP]']

# 所以不能简单直接用分词器，得写个函数处理一下
def tokenize_and_align_labels(examples):
    # 一个examples是一条样本，如同上面那种形式的
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):  # 每次处理[0,0,7,8]这样的tag里面的一个ids
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # 这个方法是把将标记映射到它们各自的单词，具体请见下面
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # 分词器处理后的ids
            if word_idx is None:  # [CLS][SEP]这种会被映射为None
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # 分词后属于一个词的部分，word_ids之后会是一样都是原来的词语
                label_ids.append(label[word_idx])  #
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx # 关键
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
