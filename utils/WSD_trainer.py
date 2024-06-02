# -*- coding: utf-8 -*-
import logging
import os
import time
import numpy as np
import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import logging
import os
import json
from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from utils.WSD_data_collator import max_seq_length, pad_sequence

class Trainer(object):
    """
    Trainer with `train` and `evaluate` functions.
    """
    def __init__(self,
            model, 
            train_loader,
            dev_loader,
            log_dir, 
            log_steps,
            validate_steps,
            num_epochs,
            lr, 
            tokenizer,
            args,
            warmup_ratio=0.1, 
            weight_decay=0.01, 
            max_grad_norm=1.0,
            gradient_accumulation_steps=0,
            early_stop_max_steps=10,
        ): # __init__函数初始化了trainer的一些参数，比如模型，训练数据，验证数据，log_dir，log_steps，validate_steps，num_epochs，lr，tokenizer，args，warmup_ratio，weight_decay，max_grad_norm，gradient_accumulation_steps，early_stop_max_steps
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.validate_steps = validate_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.tokenizer = tokenizer
        self.args = args
        self.early_stop_max_steps = early_stop_max_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if train_loader is not None:
            total_steps = len(train_loader) * self.num_epochs
            # 可以不用管它是什么功能，以后你们在别的任务里，直接这么用优化器就行
            self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                num_warmup_steps=self.warmup_ratio * total_steps, num_training_steps=total_steps)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train(self):
        # trainable_params = [n for n, p in self.model.named_parameters() if p.requires_grad]
        # logging.info ("Trainable parameters:" + ", ".join(trainable_params))
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info ("Total parameters: {}\tTrainable parameters: {}".format(total_num, trainable_num))

        logging.info ("Total batches per epoch : {}".format(len(self.train_loader)))

        best_model_store_path = os.path.join(self.log_dir, "best_model.bin")
        trained_steps = 0
        valid_loss = []
        best_acc, best_precision, best_recall, best_f1 = 0, 0, 0, 0
        early_stop_steps = 0 # early_stop_max_steps个validation内，如果没有提升，则停止训练
        # early_stop_steps是用来判断是否提前停止训练的，如果early_stop_steps >= early_stop_max_steps，那么就停止训练

        for epoch in range(self.num_epochs):
            logging.info ("\nEpoch {} Start:".format(epoch + 1))
            if early_stop_steps >= self.early_stop_max_steps:
                break
            for batch_step, inputs in enumerate(self.train_loader):
                # inputs就是 data collator的collated_batch的东西
                self.model.train()
                trained_steps += 1
                
                model_output = self.model(inputs)
                loss = model_output["loss"]
                output = model_output["output"]
                valid_loss.append(loss.item())
                
                # 这是梯度回传，可以不用管，以后都这么写就好
                loss.backward()
                if self.max_grad_norm > 0:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                # 经过log step步打印出loss
                if trained_steps > 0 and trained_steps % self.log_steps == 0:
                    avg_valid_loss = np.mean(valid_loss)
                    log_key = "Batch Step: {}\tloss: {:.3f}"
                    log_value = (trained_steps, avg_valid_loss.item(),)
                    logging.info (log_key.format(*log_value))
                    valid_loss = []
                
                # Validation
                if trained_steps > 0 and trained_steps % self.validate_steps == 0:
                    logging.info ("Evaluating...")
                    eval_output_dict = self.evaluate(loader=self.dev_loader)
                    logging.info ("Epoch {} Trained Step {} -- Average Loss: {:.3f}".format(
                        epoch + 1, trained_steps, eval_output_dict["avg_loss"])
                    )
                    if eval_output_dict["F1"] > best_f1: # 可以根据自己的需求，改成其他的指标，可以用accuracy，也可以用precision，recall，f1
                        early_stop_steps = 0
                        best_f1 = eval_output_dict["F1"]
                        best_acc = eval_output_dict["Accuracy"]
                        best_precision = eval_output_dict["Precision"]
                        best_recall = eval_output_dict["Recall"]
                        logging.info ("Epoch {} Trained Step {} -- Best F1: {:.3f} Acc: {:.3f} Precision: {:.3f} Recall: {:.3f}".format(epoch + 1, trained_steps, best_f1, best_acc, best_precision, best_recall))
                        torch.save(self.model.state_dict(), best_model_store_path)
                        logging.info ("Saved to [%s]" % best_model_store_path)
                    else:
                        early_stop_steps += 1
                if early_stop_steps >= self.early_stop_max_steps:
                    logging.info ("Early stop at Epoch {} Trained Step {}".format(epoch + 1, trained_steps))
                    break
    

    def evaluate(self, loader):
        self.model.eval()
        valid_loss = []
        label_list, pred_list = [], []
        
        with torch.no_grad():
            for inputs in loader:
                model_output = self.model(inputs)
                labels = inputs["label"]
                loss = model_output["loss"]
                logits = model_output["output"] # logits是模型的输出，也就是模型的预测结果，这里是两个值，分别是属于该类的概率和不属于该类的概率
                pred_labels = torch.argmax(logits, dim=1) # 这里是取概率最大的那个值，也就是预测的结果
                valid_loss.append(loss.item())
                
                label_list.extend(labels.tolist())
                pred_list.extend(pred_labels.tolist())
                
            avg_valid_loss = np.mean(valid_loss)
        
        # Calculate metrics, Acc, Recall, Precision, F1
        Accuracy = (np.array(label_list) == np.array(pred_list)).mean()
        TP, FP, TN, FN = 0, 0, 0, 0
        for label, pred in zip(label_list, pred_list):
            if label == 1 and pred == 1:
                TP += 1
            elif label == 1 and pred == 0:
                FN += 1
            elif label == 0 and pred == 1:
                FP += 1
            elif label == 0 and pred == 0:
                TN += 1
        Recall = TP / (TP + FN)
        Precision = TP / (TP + FP)
        F1 = 2 * Recall * Precision / (Recall + Precision)
        logging.info ("Evaluation Accuracy: {:.3f}\tRecall: {:.3f}\tPrecision: {:.3f}\tF1: {:.3f}".format(
            Accuracy, Recall, Precision, F1
        ))
        result = {
            "avg_loss": avg_valid_loss,
            "Accuracy": Accuracy,
            "Recall": Recall,
            "Precision": Precision,
            "F1": F1,
        }
        return result

    def _parse_input(self, sample: dict):
        # _parse_input函数的作用是把数据转成一个list，这样就可以用pickle来存储数据了
        return_input_ids, return_attention_mask, return_token_type_ids, return_labels = [], [], [], []
        input_ids = self.tokenizer(sample["sentence"], padding=True, truncation=True, return_tensors="pt")["input_ids"][0].tolist()
        for idx, gloss in enumerate(sample["glosses"]):
            temp_input_ids = input_ids.copy() + self.tokenizer(gloss, padding=True, truncation=True, return_tensors="pt")["input_ids"][0].tolist()[1:]
            temp_mask_ids  = [1] * len(temp_input_ids)
            temp_type_ids  = [0] * len(input_ids) + [1] * (len(temp_input_ids) - len(input_ids))
            return_input_ids.append(temp_input_ids)
            return_attention_mask.append(temp_mask_ids)
            return_token_type_ids.append(temp_type_ids)
            if idx in sample["targets"]:
                return_labels.append(1)
            else:
                return_labels.append(0)
        
        assert len(return_input_ids) == len(return_attention_mask) == len(return_token_type_ids) == len(return_labels)
        
        return return_input_ids, return_attention_mask, return_token_type_ids, return_labels # return_input_ids是一个list，里面的每个元素是一个list，这个list里面的元素是input_ids，return_attention_mask是一个list，里面的每个元素是一个list，这个list里面的元素是attention_mask，return_token_type_ids是一个list，里面的每个元素是一个list，这个list里面的元素是token_type_ids，return_labels是一个list，里面的每个元素是一个int，这个int是label
    # 这个函数的作用是把数据转成一个list，这样就可以用pickle来存储数据了
    
    def _list_to_tensor(self, list_l):
        # _list_to_tensor函数的作用是把不同长度的list变成相同长度的list，再转成tensor
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.tokenizer.pad_token_id))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.tokenizer.pad_token_id, 0).masked_fill(data_tensor != self.tokenizer.pad_token_id, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    def evaluate_per_sentence(self, test_file_path): # 传入json格式的文件
        self.model.eval()
        with open(test_file_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        Accuracy = 0.0
        with torch.no_grad():
            for sample in tqdm(test_data):
                input_ids_list, attention_mask_list, token_type_ids_list, labels_list = self._parse_input(sample)
                input_ids = self._list_to_tensor(input_ids_list)
                input_mask = self.get_attention_mask(input_ids)
                input_type_ids = self._list_to_tensor(token_type_ids_list)
                labels = torch.tensor(labels_list, dtype=torch.long).to(self.device).contiguous()
                
                batch = {
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "token_type_ids": input_type_ids,
                    "label": labels,
                }
                model_output = self.model(batch)
                pred_labels = torch.argmax(model_output["output"], dim=1).tolist()  # 【0 0 1 0 0】
                pred_labels_index = [idx for idx, label in enumerate(pred_labels) if label == 1] # 【2】
                
                sample["pred_labels"] = pred_labels_index
                
                # logging.info ("Sample: {}".format(sample))
                
                if set(pred_labels_index) | set(sample["targets"]) == set(sample["targets"]): # 这里表示，预测的结果是targets结果的子集就算正确
                    # 比如说 pred label是【2,3】，target label（就是真实的label)是【2,3,4,5】 那么就算正确 【2 3】和【2,3,4,5】取合集，那结果就是【2,3,4,5】，跟target一样
                    # 比如说 pred label是【2,3】，target label（就是真实的label)是【3,4,5】 那么就算错误  【2 3】和【3,4,5】取合集，那结果就是【2,3,4,5】，跟target不一样
                    Accuracy += 1
        
        Accuracy = Accuracy / len(test_data)
        logging.info ("Accuracy: {:.3f}".format(Accuracy))
        
    def single_sentence_predition(self, sentence, glosses):
        self.model.eval()
        
        with torch.no_grad():
            sample = {
                "sentence": sentence,
                "glosses": glosses,
                "targets": [0]
            }
            input_ids_list, attention_mask_list, token_type_ids_list, labels_list = self._parse_input(sample)
            input_ids = self._list_to_tensor(input_ids_list)
            input_mask = self.get_attention_mask(input_ids)
            input_type_ids = self._list_to_tensor(token_type_ids_list)
            labels = torch.tensor(labels_list, dtype=torch.long).to(self.device).contiguous()
            
            batch = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "token_type_ids": input_type_ids,
                "label": labels,
            }
            model_output = self.model(batch)
            pred_labels = torch.argmax(model_output["output"], dim=1).tolist()  # 【0 0 1 0 0】
            pred_labels_index = [idx for idx, label in enumerate(pred_labels) if label == 1] # 【2】
            
        return pred_labels_index