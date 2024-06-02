import torch
import nltk
import ast
import json
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import argparse
import logging
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec

TGT = "[TGT]"

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, output, target):
        # 计算余弦相似度
        cos_sim = self.cosine_similarity(output, target)
        # 1 - 余弦相似度 -> 转化为最小化问题
        loss = 1 - cos_sim.mean()
        return loss
    
class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return False

    def save_checkpoint(self, val_loss, model):
        logging.info(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')

# 定义一个字典用于保存词向量
def get_vector_perword(target, sense_key, concept_vectorizer):
    target_list = ast.literal_eval(target)
    sense_key = ast.literal_eval(sense_key)
    sense = sense_key[target_list[0]]
    synset = wn.lemma_from_key(sense).synset()
    synset_name = synset.name()
    concept_vector = concept_vectorizer.get_sig_vector(concept_id=synset_name)
    return sense, concept_vector

def get_all_sense_vector(df_here, concept_vectorizer): #这部分太慢了，可以用多线程
    Vector_defination = {}
    Sense_list = df_here["sense_keys"].values.tolist()
    Target_list = df_here["targets"].values.tolist()

    for index in tqdm(range(len(Sense_list)), desc="Processing", unit="items"):
        sense_key, vector = get_vector_perword(Target_list[index], Sense_list[index], concept_vectorizer)
        Vector_defination[sense_key] = vector
    return Vector_defination

def handle_not_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, default=handle_not_serializable)
    logging.info ("Save file successfully!")

def read_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_synset2vector():
    synset2vector = Word2Vec.load('./ckpt/synset_node2vec.model')
    return synset2vector

def target_extractor(data):
    targets = set()
    for sense_keys in tqdm(data["sense_keys"]):
        sense_keys = ast.literal_eval(sense_keys)
        for key in list(sense_keys):
            synset_word = str(wn.lemma_from_key(key).synset())[8:-2]
            targets.add(synset_word)
    return targets

def pre_detection_targets(train_data, dev_data, test_data):
    # 从数据集中提取出所有的target
    # Length of train_targets:  50145
    # Length of dev_targets:  2050
    # Length of test_targets:  10355
    # Length of not_in_dev:  15
    # Length of not_in_test:  0
    # 结果表明，在dev和test中，有一些target不在train中，但是数量很少，就15个，所以可以忽略不计
    train_targets = target_extractor(train_data)
    dev_targets = target_extractor(dev_data)
    test_targets = target_extractor(test_data)
    logging.info ("Length of train_targets: {}".format(len(train_targets)))
    logging.info ("Length of dev_targets: {}".format(len(dev_targets)))
    logging.info ("Length of test_targets: {}".format(len(test_targets)))
    
    not_in_dev, not_in_test = [], []
    for target in dev_targets:
        if target not in train_targets:
            not_in_dev.append(target)
        if target not in test_targets:
            not_in_test.append(target)
    logging.info ("Length of not_in_dev: {}".format(len(not_in_dev)))
    logging.info ("Length of not_in_test: {}".format(len(not_in_test)))
    
def read_data():
    Train_WSD_db = pd.read_csv("./dataset/train_wsd.csv")
    Dev_WSD_db = pd.read_csv("./dataset/dev_wsd.csv")
    Test_WSD_db = pd.read_csv("./dataset/all.csv")
    logging.info ("Length of Train_WSD_db: {}".format(len(Train_WSD_db)))
    logging.info ("Length of Dev_WSD_db: {}".format(len(Dev_WSD_db)))
    logging.info ("Length of Test_WSD_db: {}".format(len(Test_WSD_db)))
    return Train_WSD_db, Dev_WSD_db, Test_WSD_db

def nltk_download_datasets():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('tagsets')
    nltk.download('stopwords')
    
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")