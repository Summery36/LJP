# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 20:14
# @Author  : Giraffe
import json


def data_prepare(train_path, valid_path, test_path,  article_path, accusation_path):
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for i in f.readlines():
            train_data.append(json.loads(i))
    valid_data = []
    with open(valid_path, 'r', encoding='utf-8') as f:
        for i in f.readlines():
            valid_data.append(json.loads(i))
    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for i in f.readlines():
            test_data.append(json.loads(i))

    article_to_idx = json.load(open(article_path, 'r', encoding='utf-8'))
    accusation_to_idx = json.load(open(accusation_path, 'r', encoding='utf-8'))
    return train_data, valid_data, test_data, article_to_idx, accusation_to_idx
