# -*- coding: utf-8 -*-
# 把数据处理的部分放在一个类里，这样更清晰
# 而且可以把数据处理的部分放在GPU上，这样更快
# 而且可以异步处理数据，加速训练
import logging
import os
import json
import pickle
import random
import dataclasses
import ast
import logging
import pandas as pd
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset
from tqdm import tqdm
from nltk.corpus import wordnet as wn


@dataclass(frozen=True)
class InputFeature:
    # InputFeature的作用是把数据转成一个dict，这样就可以用json来存储数据了
    """
    A single set of features of data.
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label: int
    
    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

class WSDDataset(Dataset):
    # WSDDataset的作用是把数据转成一个list，这样就可以用pickle来存储数据了
    # pickle是python自带的序列化工具，可以把python的数据结构转成二进制文件，这样就可以存储到硬盘上了
    """
    Self-defined WSD Dataset class.
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,
        data_path,
        data_partition,
        tokenizer,
        cache_dir=None,
        is_test=False,
    ): # __init__函数初始化了dataset的一些参数，比如数据的路径，数据的分区，tokenizer，cache_dir，is_test
        # 其中data_partition是指train, dev, test
        # tokenizer是指bert的tokenizer
        self.data_partition = data_partition
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.is_test = is_test
        
        self.instances = [] # 这个instances就是一个list，里面存储了所有的数据
        self._cache_instances(data_path) # _cache_instances的作用是把数据转成一个list，这样就可以用pickle来存储数据了
    
    def _cache_instances(self, data_path):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        signature = "{}_cache.pkl".format(self.data_partition) # 这个signature的作用是给cache的文件起一个名字，比如train_cache.pkl；format的作用是把self.data_partition的值填到{}里面
        if self.cache_dir is not None:
            # 这个if语句判断cache_dir是否存在，如果不存在就创建一个
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, signature)
        else:
            cache_dir = os.mkdir("caches")
            cache_path = os.path.join(cache_dir, signature)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                logging.info ("Loading cached instances from {}".format(cache_path))
                self.instances = pickle.load(f)
        else:          
            logging.info ("Loading raw data from {}".format(data_path))
            all_samples = []
            data_db = pd.read_csv(data_path) # 读取数据
            for id, sentence, sense_keys, glosses, targets in tqdm(zip(data_db["id"], data_db["sentence"], data_db["sense_keys"], data_db["glosses"], data_db["targets"])): 
                # for循环的作用是把数据转成一个dict，这样就可以用json来存储数据了
                data_sample = {
                    "id": id,
                    "sentence": sentence,
                    "sense_keys": ast.literal_eval(sense_keys),
                    "glosses": ast.literal_eval(glosses),
                    "targets": ast.literal_eval(targets),
                }
                all_samples.append(data_sample)
            with open(cache_path.replace(".pkl", ".json"), 'w') as f:
                # 这里pkl的作用是把数据存成二进制文件，这样可以加快读取速度
                # 将pkl文件转成json文件的作用是可以看到数据的样子
                # 把数据存成json文件，这样可以看到数据的样子
                json.dump(all_samples, f, indent=4)
                # indent=4的作用是让json文件的格式更好看一些
            
            logging.info ("Creating cache instances {}".format(signature))
            
            # 逐句测试的时候才会使用到这个函数
            for sample in tqdm(all_samples):
                input_ids_list, attention_mask_list, token_type_ids_list, labels_list = self._parse_input(sample)
                for input_ids, attention_mask, token_type_ids, label in zip(input_ids_list, attention_mask_list, token_type_ids_list, labels_list):
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                        "label": label,
                    }
                    feature = InputFeature(**inputs)
                    self.instances.append(feature)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info ("Total of {} instances were cached.".format(len(self.instances)))
    
    def _parse_input(self, sample: dict):
        # _parse_input的作用是把数据转成一个list，这样就可以用pickle来存储数据了
        return_input_ids, return_attention_mask, return_token_type_ids, return_labels = [], [], [], []
        input_ids = self.tokenizer(sample["sentence"], padding=True, truncation=True, return_tensors="pt")["input_ids"][0].tolist() # 原本是一个tensor，现在转成list
        for idx, gloss in enumerate(sample["glosses"]):
            temp_input_ids = input_ids.copy() + self.tokenizer(gloss, padding=True, truncation=True, return_tensors="pt")["input_ids"][0].tolist()[1:] # 因为bert的输入是[CLS] + sentence + [SEP] + gloss + [SEP]
            temp_mask_ids  = [1] * len(temp_input_ids)
            temp_type_ids  = [0] * len(input_ids) + [1] * (len(temp_input_ids) - len(input_ids)) # 区分sentence和gloss
            return_input_ids.append(temp_input_ids)
            return_attention_mask.append(temp_mask_ids)
            return_token_type_ids.append(temp_type_ids)
            if idx in sample["targets"]:
                return_labels.append(1)
            else:
                return_labels.append(0)
        
        assert len(return_input_ids) == len(return_attention_mask) == len(return_token_type_ids) == len(return_labels)
        
        return return_input_ids, return_attention_mask, return_token_type_ids, return_labels

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

