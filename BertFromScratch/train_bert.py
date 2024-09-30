# -*- coding: utf-8 -*-
# @Time    : 2023/11/28 19:02
# @Author  : Giraffe
import torch
import torch.nn as nn
import os
import warnings
import configparser
from loguru import logger
from train_tools import *
from torch.utils.data.dataloader import DataLoader
from data_prepare import data_prepare
from models import BERTModel
from transformers import Adafactor, BertTokenizer

# 读取配置文件
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")

# 获取配置参数：从配置文件中读取训练、验证和测试路径以及其他参数
section_name = 'DEFAULT'
args_list = ['train_path', 'valid_path', 'test_path', 'article_path', 'accusation_path']
train_path, valid_path, test_path, article_path, accusation_path = get_config(config, section_name, args_list)
section_name = 'BERT'
args_list = ['batch_size', 'small_sample', 'gpu', 'accum_step', 'patience', 'model_name', 'mode3.'
                                                                                          'l_savepath', 'log_path']
batch_size, small_sample, gpu, accum_step, patience, model_name, model_savepath, log_path = get_config(config, section_name, args_list)

# 设定训练参数：如果small_sample为True，则设置训练周期为2，否则从配置文件中读取
epochs = 2 if small_sample else config.getint(section_name, 'epochs')

# 选择设备：根据GPU配置和可用性选择运行设备
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

# 定义日志记录器：如果日志文件存在，则删除并创建一个新的日志记录器
if os.path.exists(log_path):
    os.remove(log_path)
logger.add(log_path)

# 数据准备：调用data_prepare函数来加载数据，并获取文章和罪名的索引映射
train_data, valid_data, test_data, article_to_idx, accusation_to_idx = \
    data_prepare(train_path, valid_path, test_path, article_path, accusation_path)

# 处理小样本数据：如果使用小样本数据，则缩小数据集的大小
if small_sample:
    small_size = batch_size * 10
    train_data, valid_data, test_data = train_data[:small_size], valid_data[:small_size], test_data[:small_size]

# 创建数据加载器：使用BertTokenizer进行分词，并创建用于训练、验证和测试的数据加载器
data_list = [train_data, valid_data, test_data]
logger.info('start tokenizing')
tokenizer = BertTokenizer.from_pretrained(model_name)
train_data, valid_data, test_data = get_dataset(data_list, article_to_idx, accusation_to_idx, tokenizer)

# print(train_data[0:4])

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

# 定义模型：创建一个BERT模型实例
model = BERTModel(model_name, 768, len(article_to_idx), len(accusation_to_idx))


# 训练函数：定义一个函数来训练BERT模型
def train_bert(train_loader, valid_loader, device, model, epochs, accum_step, logger, early_stop=None):
    model = model.to(device)
    logger.info(f'train on {device}')
    start_evaluate = 0 if small_sample else 6
    criterion = [MultiLabelSoftmaxLoss(len(article_to_idx)), nn.CrossEntropyLoss()]
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    for epoch in range(epochs):
        model.train()
        # the train detail is defined in function train_per_epoch
        train_loss = train_per_epoch(train_loader, device, model, criterion, optimizer, accum_step)
        logger.info(f'epoch:{epoch},loss:{train_loss:.4f}')

        # if current epoch >= start_evaluate, start evaluate.
        if epoch >= start_evaluate:
            model.eval()
            warnings.filterwarnings('ignore')

            # the evaluate detail is defined in function evaluate_per_epoch
            metric = evaluate_per_epoch(valid_loader, device, model)
            metric_info = ['article_acc', 'article_f1', 'article_p', 'article_r', 'accusation_acc', 'accusation_f1',
                           'accusation_p', 'accusation_r']
            result_info = f'epoch:{epoch}'
            for metric_number, m in enumerate(metric):
                result_info += f',{metric_info[metric_number]}:{m:.4f}'
            logger.info(result_info)

            # whether early_stop or not.
            if early_stop is not None:
                stop = early_stop(metric[1], model)
                if stop:
                    logger.info('Early stopping!')
                    break
    return


# if you use cuda, you can use empty_cache
# torch.cuda.empty_cache()

train_bert(train_loader, valid_loader, device, model, epochs, accum_step, logger,
           early_stop=EarlyStopping(model_savepath, logger, patience))


# 定义测试函数
def test_bert(test_loader, device, model, logger):
    model = model.to(device)
    model.eval()

    metric = evaluate_per_epoch(test_loader, device, model)
    metric_info = ['article_acc', 'article_f1', 'article_p', 'article_r', 'accusation_acc', 'accusation_f1',
                   'accusation_p', 'accusation_r']
    test_info = f'Test_metric'
    for metric_number, m in enumerate(metric):
        test_info += f',{metric_info[metric_number]}:{m:.4f}'
    logger.info(test_info)
    return

# 训练和测试模型：调用train_bert函数来训练模型，在训练完成后加载模型参数，使用test_bert函数进行测试
model.load_state_dict(torch.load(model_savepath))
test_bert(test_loader=test_loader, device=device, model=model, logger=logger)
