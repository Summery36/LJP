import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataset import BERTDataset


class EarlyStopping:
    def __init__(self, output_path, logger, patience=3, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.output_path = output_path
        self.logger = logger

    def __call__(self, val_acc, model):
        score = val_acc
        # initial best_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score <= self.best_score + self.delta:
            # if current score <= best_score, early_stop counter+1, when counter == patience, then stop training.
            self.counter += 1
            self.logger.info(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # get new best_score, then refresh best_score and save the current model.
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.output_path)


# CrossEntropyLoss for multi-label task.
class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self, task_num=0):
        super(MultiLabelSoftmaxLoss, self).__init__()
        self.task_num = task_num
        self.criterion = []
        for a in range(0, self.task_num):
            self.criterion.append(nn.CrossEntropyLoss())

    def forward(self, outputs, labels):
        # outputs->[batch_size,article_num,2]  labels->[batch_size,article]
        loss = 0
        for a in range(0, outputs.shape[1]):
            # output->[batch_size,2] labels[:,a]->[batch_size]
            output = outputs[:, a, :].view(outputs.shape[0], -1)
            loss += self.criterion[a](output, labels[:, a])
        return loss


def calculate_loss(output, target, criterion):
    article_output, article_target, article_criterion = output[0], target[0], criterion[0]
    accusation_output, accusation_target, accusation_criterion = output[1], target[1], criterion[1]
    article_loss = article_criterion(article_output, article_target)
    accusation_loss = accusation_criterion(accusation_output, accusation_target)

    return (article_loss + accusation_loss) / 2


def calculate_metric(target, pred):
    acc = accuracy_score(target, pred)
    f1 = f1_score(target, pred, average='macro')
    precision = precision_score(target, pred, average='macro')
    recall = recall_score(target, pred, average='macro')
    return [acc, f1, precision, recall]


def train_per_epoch(train_loader, device, model, criterion, opt, accum_step=1):
    loss_sum = 0
    for batch_number, batch in enumerate(tqdm(train_loader)):
        inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
        target = [batch['article'].to(device), batch['accusation'].to(device)]
        output = model(inputs)
        batch_loss = calculate_loss(output, target, criterion)
        loss_sum += batch_loss.item()
        batch_loss = batch_loss / accum_step
        batch_loss.backward()
        # in Pytorch, loss.backward() will accumulate gradient.
        # gradient will accumulate 'accum_step' batches, it is efficient when batch_size is tiny.
        if (batch_number + 1) % accum_step == 0 or (batch_number + 1) == len(train_loader):
            opt.step()
            opt.zero_grad()

    return loss_sum / len(train_loader)


def evaluate_per_epoch(valid_loader, device, model):
    article_predicts, accusation_predicts = [], []
    article, accusation = [], []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
            predicts = model(inputs)
            #[0,1,0,0,0,1,0,1]
            # get results of article predicts, predicts[0]:[batch_size,article_numbers,2] -> [batch_size,article_numbers].
            article_predicts.extend((torch.softmax(predicts[0], dim=-1)[:, :, -1] > 0.5).long().cpu().tolist())
            # get results of accusation predicts, predicts[1]:[batch_size,accusation_numbers] -> [batch_size,1].
            # please consider what is differences between two tasks.
            _, index = torch.max(predicts[1], dim=-1)
            accusation_predicts.extend(index.cpu().tolist())

            article.extend(batch['article'].tolist())
            accusation.extend(batch['accusation'].tolist())

    article_metric = calculate_metric(article, article_predicts)
    accusation_metric = calculate_metric(accusation, accusation_predicts)
    metric = article_metric + accusation_metric
    return metric


def get_config(config, section, args_list):
    args = []
    for args_name in args_list:
        try:
            args.append(config.getint(section, args_name))
        except:
            args.append(config.get(section, args_name))
    return args


def get_dataset(data_list, article_to_idx, accusation_to_idx, tokenizer):
    all_data = []
    for data in data_list:
        all_data.append(BERTDataset(data, article_to_idx, accusation_to_idx, tokenizer))

    return all_data
