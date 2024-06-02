# 这个模型是基于Bert的模型，用于做WSD任务
# Accuracy:     0.816
# F1:           0.489
# Precision:    0.484
# Recall:       0.494
# Evaluation Per Sentence: Accuracy: 0.642
import math
import torch
import logging
import torch.nn as nn
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer

class WSDModel(nn.Module):
    def __init__(self, tokenizer=None):
        super(WSDModel, self).__init__()
        self.tokenizer = tokenizer
        # tokenizer的作用是将输入的文本转换成ids，然后输入到模型中
        # 其中ids包括input_ids, attention_mask, token_type_ids，他们分别是输入的文本，mask，和segment
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # bert-base-uncased的hidden_size是768，bert-large-uncased的hidden_size是1024，下一步优化可以考虑使用bert-large-uncased
        self.bert.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(0.2)
        self.ranking_linear = nn.Linear(768, 2) # 768是bert-base-uncased的hidden_size, 1024是bert-large-uncased的hidden_size

    def forward(self, inputs):
        input_ids, attention_mask, token_type_ids, label = inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["label"]
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs[1]) # [CLS] tokens embedding
        # CLS token的作用是用于分类任务，它的输出是一个向量，这个向量可以用于分类任务
        # pooled_output的shape是[batch_size, hidden_size]
        logits = self.ranking_linear(pooled_output) # 输出的是两个值，分别是属于该类的概率和不属于该类的概率
        # ranking_linear的作用是将pooled_output映射到两个值，分别是属于该类的概率和不属于该类的概率
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, label)
        output_dict = {
            "loss": loss,
            "output": logits,
        }
        return output_dict