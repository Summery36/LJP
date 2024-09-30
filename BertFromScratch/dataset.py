# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 20:22
# @Author  : Giraffe

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class BERTDataset(Dataset):
    def __init__(self, data, article_to_idx, accusation_to_idx, tokenizer):
        self.data = data
        self.length = len(data)
        self.tokenizer = tokenizer
        self.fact = []
        self.article = []
        self.accusation = []
        self.input_ids = []
        self.attention_mask = []

        for i in data:
            self.fact.append(i['fact'])
            # convert origin article list into 0 or 1 list
            # [1,0,1,1,0,0,0,1,1,……]
            article_zero = [0 for _ in range(len(article_to_idx))]
            for j in i['meta']['relevant_articles']:
                article_zero[article_to_idx[str(j)]] = 1
            self.article.append(article_zero)

            # convert origin accusation into accusation_index
            self.accusation.append(accusation_to_idx[i['meta']['accusation'][0]])

        for i in tqdm(range(0, self.length, 1000)):
            tokenize_texts = self.tokenizer(self.fact[i:min(
                i + 1000, self.length)], padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            self.input_ids.append(tokenize_texts['input_ids'])
            self.attention_mask.append(tokenize_texts['attention_mask'])

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_mask = torch.cat(self.attention_mask, dim=0)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # return a sample of all data
        return {'input_ids': self.input_ids[item], 'attention_mask': self.attention_mask[item],
                'article': torch.tensor(self.article[item], dtype=torch.long),
                'accusation': torch.tensor(self.accusation[item], dtype=torch.long)}


