# -*- coding: utf-8 -*-
import torch
import numpy as np
import logging

def max_seq_length(list_l):
    # 这个函数用于计算一个list中最长的那个list的长度，作为padding的长度
    return max(len(l) for l in list_l)

def pad_sequence(list_l, max_len, padding_value=0):
    # 这个函数用于把一个list变成一个长度为max_len的list，如果这个list的长度小于max_len，那么就用padding_value来填充
    if len(list_l) <= max_len:
        padding_l = [padding_value] * (max_len - len(list_l))
        padded_list = list_l + padding_l
    else:
        padded_list = list_l[:max_len]
    return padded_list



class DataCollator(object):
    # 这个函数的作用是把不同长度的list变成相同长度的list，再转成tensor
    """
    Data collator for BertWordVecPredictionModel
    """
    def __init__(self, device, padding_idx=0): # 参数device是用于指定使用cpu还是gpu 参数padding_idx是用于指定padding的值
        self.device = device
        self.padding_idx = padding_idx
    
    def list_to_tensor(self, list_l): # 这个函数的作用就是把不同长度的list变成相同长度的list，再转成tensor
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.padding_idx))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.padding_idx, 0).masked_fill(data_tensor != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    def custom_collate(self, mini_batch):  # 最重要的就是这个函数
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = [], [], [], []
        
        for sample in mini_batch:
            batch_input_ids.append(sample.input_ids)
            batch_attention_mask.append(sample.attention_mask)
            batch_token_type_ids.append(sample.token_type_ids)
            batch_labels.append(sample.label)
            
        # inputs
        input_ids = self.list_to_tensor(batch_input_ids)
        input_mask = self.get_attention_mask(input_ids)
        input_type_ids = self.list_to_tensor(batch_token_type_ids)
        labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device).contiguous()
        
        collated_batch = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": input_type_ids,
            "label": labels,
        }
        return collated_batch
# 这个python文件的作用是把数据变成batch，然后把batch变成tensor
# 整个project的目的是把一个句子和一个词的embedding拼接起来，然后再做一个regression，这个regression的目的是让这个拼接起来的embedding和这个词的embedding尽可能的相似