# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "模型"
__Created__ = 2023/12/14 14:50
"""
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, model_path, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
