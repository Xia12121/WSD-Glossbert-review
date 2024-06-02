# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import torch
import nltk
import torch.nn as nn
import tensorflow as tf
import ast
import logging
import sys
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors, Word2Vec
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorflow.keras.layers import Layer, Dot, Activation
from node2vec import Node2Vec
from utils.utils import (
    get_all_sense_vector,
    save_file,
    load_synset2vector,
    pre_detection_targets,
    read_data,
    set_random_seed,
    str2bool,
)
from utils.Your_dataset import WSDDataset
from utils.Your_data_collator import DataCollator
from utils.WSD_trainer import Trainer
from model.Your_WSDModel import WSDModel
from model.ConceptVectorizer import ConceptVectorizer

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout)
    ]
)

def test_lemma_sample():
    # 通过 sense key 查询 Synset 对象
    synset1 = wn.lemma_from_key('long%5:00:00:unsound:00').synset()
    synset2 = wn.lemma_from_key('objective%1:06:00::').synset()

    # 打印 Synset 对象，以便查看概念信息
    print(synset1, synset1.lemmas()[0].name())
    print(synset2, synset2.examples())

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_data_path', type=str, default='./dataset/train_refined.csv')
    parser.add_argument('--dev_data_path', type=str, default='./dataset/dev_refined.csv')
    parser.add_argument('--test_data_path', type=str, default='./dataset/test_refined.csv')
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--validate_steps', type=int, default=2000)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--node2vec_model', type=str, default='./synset_ckpt/synset_node2vec.model')
    return parser.parse_args()

if __name__ == "__main__":
    # 设置超参数
    args = parse_config()
    
    # 设置随机种子
    set_random_seed(args.random_seed)# 随机种子的作用是保证在不同GPU上的结果一致
    
    # 设置device, tokenizer, model, dataset, data collator
    vectorizer = Word2Vec.load(args.node2vec_model).wv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = WSDModel(tokenizer=tokenizer)
    model.to(device)
    train_dataset = WSDDataset(
        data_path=args.train_data_path, data_partition="train", 
        tokenizer=tokenizer, cache_dir=args.cache_dir, is_test=False,
    )
    dev_dataset = WSDDataset(
        data_path=args.dev_data_path, data_partition="dev", 
        tokenizer=tokenizer, cache_dir=args.cache_dir, is_test=False,
    )
    test_dataset = WSDDataset(
        data_path=args.test_data_path, data_partition="test", 
        tokenizer=tokenizer, cache_dir=args.cache_dir, is_test=True,
    )
    collator = DataCollator(device=device, padding_idx=tokenizer.pad_token_id, target_vectorizer=vectorizer)
    
    # 加载dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator.custom_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.custom_collate)
    
    trainer = Trainer(model=model, train_loader=train_loader, dev_loader=dev_loader, log_dir=args.log_dir, log_steps=args.log_steps, num_epochs=args.num_epochs, lr=args.lr, validate_steps=args.validate_steps, warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, tokenizer=tokenizer, args=args)
    
    
    trainer.train()