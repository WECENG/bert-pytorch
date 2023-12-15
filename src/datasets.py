# -*- coding: UTF-8 -*-
"""
__Author__ = "WECENG"
__Version__ = "1.0.0"
__Description__ = "数据集"
__Created__ = 2023/12/14 14:52
"""
import numpy as np
import torch.utils.data

from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datas, model_path):
        self.labels = datas['label']
        tokenizer = BertTokenizer.from_pretrained(model_path)
        self.reviews = [
            tokenizer(str(review), padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            for review in datas['review']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        """
        默认情况下，DataLoader 的 collate_fn 使用 torch.utils.data._utils.collate.default_collate，
        这个函数要求 batch 中的每个元素都是 PyTorch 的 tensor、numpy array、数字、字典或列表。
        """
        return self.reviews[item], np.array(self.labels[item])
