#  直接 Python3 WSD_BertBased.py 就可以运行了，参数在下面的parse_config()函数里面设置

# 函数，函数的输入是【sentence，gloss（list）】输出就是目标词属于哪个义项【list】
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ast
import sys
import logging
import os
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors, Word2Vec
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from tqdm import tqdm
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
from model.WSD_BertBasedModel import WSDModel
from model.BiEncoder import BiEncoder
from model.CrossEncoder import CrossEncoder
from model.PolyEncoder import PolyEncoder
from utils.utils import TGT

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout)
    ]
) # 设置日志格式

def parse_config(): # parse_config函数的作用是设置超参数
    parser = argparse.ArgumentParser()
    # 以下是模型参数
    parser.add_argument('--mode', type=str, default='pred_single', choices=['train', 'test', 'pred_single']) # 这里的mode是指train还是test,pred_single
    parser.add_argument('--model_name', type=str, default='WSD_BertBased', choices=['WSD_BertBased', 'BiEncoder', 'PolyEncoder', 'CrossEncoder']) # 这里的model_name是指WSD_BertBased还是BiEncoder还是PolyEncoder
    parser.add_argument('--train_data_path', type=str, default='./dataset/train_wsd.csv') # 因为我是把数据集编码完之后会存在cache文件里，如果你更改了数据，那需要把cache文件夹里对应的文件删除
    parser.add_argument('--dev_data_path', type=str, default='./dataset/all.csv')
    parser.add_argument('--test_data_path', type=str, default='./dataset/all.csv') # 如果你更换了数据，那需要把cache里面对应的文件删除
    # 以下是训练参数，训练参数是可以调的
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=48) # 尽量大一点，但是不要超过显存限制
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--validate_steps', type=int, default=400)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--early_stop_max_steps', type=int, default=10) # 10次验证集的Metric没有上升就停止训练
    parser.add_argument('--node2vec_model', type=str, default='./synset_ckpt/synset_node2vec.model')
    parser.add_argument('--poly_m', type=int, default=10) # PolyEncoder的m值
    return parser.parse_args()

def train_model(args):
    # 设置随机种子
    set_random_seed(args.random_seed)
    
    # 设置device, tokenizer, model, dataset, data collator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    logging.info ("Before adding special tokens, the length of tokenizer is: {}".format(len(tokenizer)))
    if TGT not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [TGT]})
    logging.info ("After adding special tokens, the length of tokenizer is: {}".format(len(tokenizer)))
    
    if args.model_name == "WSD_BertBased":
        model = WSDModel(tokenizer=tokenizer) # 这个WSDModel的作用是把bert的输出映射到两个值，分别是属于该类的概率和不属于该类的概率
        from utils.WSD_dataset import WSDDataset # 这个WSDDataset的作用是把数据转成一个list，这样就可以用pickle来存储数据了
        from utils.WSD_data_collator import DataCollator # 这个DataCollator的作用是把不同长度的list变成相同长度的list，再转成tensor
        from utils.WSD_trainer import Trainer
    elif args.model_name == "BiEncoder":
        model = BiEncoder(tokenizer=tokenizer)
        from utils.BiEncoder_dataset import WSDDataset
        from utils.BiEncoder_data_collator import DataCollator
        from utils.BiEncoder_trainer import Trainer
    elif args.model_name == "CrossEncoder":
        model = CrossEncoder(tokenizer=tokenizer)
        from utils.CrossEncoder_dataset import WSDDataset
        from utils.CrossEncoder_data_collator import DataCollator
        from utils.CrossEncoder_trainer import Trainer
    elif args.model_name == "PolyEncoder":
        model = PolyEncoder(tokenizer=tokenizer, poly_m=args.poly_m)
        from utils.PolyEncoder_dataset import WSDDataset
        from utils.PolyEncoder_data_collator import DataCollator
        from utils.PolyEncoder_trainer import Trainer
    model.to(device)
    
    train_dataset = WSDDataset(
        data_path=args.train_data_path, data_partition="train",  # 这里的data_partition是指train, dev, test
        tokenizer=tokenizer, cache_dir=args.cache_dir+'/'+args.model_name, is_test=False, # 这里的is_test是指是否是test，如果是test，那么就不需要label
    )
    dev_dataset = WSDDataset(
        data_path=args.dev_data_path, data_partition="dev", # 这里的data_partition是指train, dev, test
        tokenizer=tokenizer, cache_dir=args.cache_dir+'/'+args.model_name, is_test=True, # 这里的is_test是指是否是test，如果是test，那么就不需要label
    )
    test_dataset = WSDDataset(
        data_path=args.test_data_path, data_partition="test", 
        tokenizer=tokenizer, cache_dir=args.cache_dir+'/'+args.model_name, is_test=True,
    )
    collator = DataCollator(device=device, padding_idx=tokenizer.pad_token_id)
    
    # 加载dataloader，可以让cpu和gpu同时跑，异步处理，不需要等待
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator.custom_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.custom_collate)
    # DataLoader的作用是把数据转成一个list，这样就可以用pickle来存储数据了

    trainer = Trainer(model=model, train_loader=train_loader, dev_loader=dev_loader, log_dir=args.log_dir+'/'+args.model_name, log_steps=args.log_steps, num_epochs=args.num_epochs, lr=args.lr, validate_steps=args.validate_steps, warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, tokenizer=tokenizer, args=args, early_stop_max_steps=args.early_stop_max_steps)
    trainer.train()

def test_model(args):
    # 设置随机种子
    set_random_seed(args.random_seed)
    
    # 设置device, tokenizer, model, dataset, data collator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    logging.info ("Before adding special tokens, the length of tokenizer is: {}".format(len(tokenizer)))
    if TGT not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [TGT]})
    logging.info ("After adding special tokens, the length of tokenizer is: {}".format(len(tokenizer)))
    
    if args.model_name == "WSD_BertBased":
        model = WSDModel(tokenizer=tokenizer)
        from utils.WSD_dataset import WSDDataset
        from utils.WSD_data_collator import DataCollator
        from utils.WSD_trainer import Trainer
    elif args.model_name == "BiEncoder":
        model = BiEncoder(tokenizer=tokenizer)
        from utils.BiEncoder_dataset import WSDDataset
        from utils.BiEncoder_data_collator import DataCollator
        from utils.BiEncoder_trainer import Trainer
    elif args.model_name == "CrossEncoder":
        model = CrossEncoder(tokenizer=tokenizer)
        from utils.CrossEncoder_dataset import WSDDataset
        from utils.CrossEncoder_data_collator import DataCollator
        from utils.CrossEncoder_trainer import Trainer        
    elif args.model_name == "PolyEncoder":
        model = PolyEncoder(tokenizer=tokenizer)
        from utils.PolyEncoder_dataset import WSDDataset
        from utils.PolyEncoder_data_collator import DataCollator
        from utils.PolyEncoder_trainer import Trainer
    model.to(device)
    
    checkpoint = args.log_dir+'/'+args.model_name+'/best_model.bin'
    if not os.path.exists(checkpoint):
        raise ValueError("Invalid checkpoint!")
    model.load_state_dict(torch.load(checkpoint))
    
    test_dataset = WSDDataset(
        data_path=args.test_data_path, data_partition="test", 
        tokenizer=tokenizer, cache_dir=args.cache_dir+'/'+args.model_name, is_test=True,
    )
    collator = DataCollator(device=device, padding_idx=tokenizer.pad_token_id)
    
    # 加载dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator.custom_collate)
    # test_loader的作用是进行异步处理
    trainer = Trainer(model=model, train_loader=None, dev_loader=None, log_dir=args.log_dir+'/'+args.model_name, log_steps=args.log_steps, num_epochs=args.num_epochs, lr=args.lr, validate_steps=args.validate_steps, warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, tokenizer=tokenizer, args=args)
    trainer.evaluate(test_loader)
    trainer.evaluate_per_sentence(os.path.join(args.cache_dir+'/'+args.model_name, "test_cache.json"))
    
def single_sentence_predition(sentence, glosses): # sentence 是一个字符串，xx xxx [TGT] 目标词 [TGT] xxxx, glosses是一个list，里面是目标词所有的义项
    # 必须设置的参数
    model_name = "WSD_BertBased"  # WSD_BertBased', 'BiEncoder', 'PolyEncoder', 'CrossEncoder'. CrossEncoder训练时间太长了，这里没有checkpoint，最好使用WSD_BertBased
    poly_m = 10  # PolyEncoder的m值, 我已经训练好了m=10的PolyEncoder，可以直接使用， 如果你想训练自己的PolyEncoder，那么这里的m值就是你训练的时候的m值
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer的作用是将输入的文本转换成ids，然后输入到模型中
    logging.info ("Before adding special tokens, the length of tokenizer is: {}".format(len(tokenizer)))
    if TGT not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [TGT]})
    logging.info ("After adding special tokens, the length of tokenizer is: {}".format(len(tokenizer)))
    
    if model_name == "WSD_BertBased":
        model = WSDModel(tokenizer=tokenizer)
        from utils.WSD_trainer import Trainer
    elif model_name == "BiEncoder":
        model = BiEncoder(tokenizer=tokenizer)
        from utils.BiEncoder_trainer import Trainer
    elif model_name == "CrossEncoder":
        model = CrossEncoder(tokenizer=tokenizer)
        from utils.CrossEncoder_trainer import Trainer        
    elif model_name == "PolyEncoder":
        model = PolyEncoder(tokenizer=tokenizer, poly_m=poly_m)
        from utils.PolyEncoder_trainer import Trainer
    else:
        raise ValueError("Invalid model name!")
    model.to(device)
    
    checkpoint = './log/'+model_name+'/best_model.bin'
    if not os.path.exists(checkpoint):
        raise ValueError("Invalid checkpoint!")
    model.load_state_dict(torch.load(checkpoint))
    
    trainer = Trainer(model=model, train_loader=None, dev_loader=None, log_dir='./log/'+model_name, log_steps=None, num_epochs=None, lr=None, validate_steps=None, warmup_ratio=None, weight_decay=None, tokenizer=tokenizer, args=None)
    
    truth_glosses = trainer.single_sentence_predition(sentence, glosses)
    
    raw_glosses = [glosses[t] for t in truth_glosses]
    # raw_glosses 是所有正确的义项
    return raw_glosses

if __name__ == "__main__":
    # 设置超参数
    args = parse_config()
    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)
    elif args.mode == "pred_single":  # 需要设置mode为pred_single, 并且必须设置model_name
        sentence = "When their changes are [TGT] completed [TGT] , and after they have worked up a sweat , ringers often skip off to the local pub , leaving worship for others below ."
        glosses = ['bring to a whole, with all the necessary parts or elements', 'complete a pass', 'come or bring to a finish or an end', 'complete or carry out', 'write all the required information onto a form']
        truth_glosses = single_sentence_predition(sentence, glosses)
        print ("The truth glosses is :", truth_glosses)
    else:
        raise ValueError("Invalid mode!")
    