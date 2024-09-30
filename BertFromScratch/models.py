import torch
import torch.nn as nn
from transformers import BertModel


class BERTModel(nn.Module):
    def __init__(self, model_name, hidden_size, articles, accusations):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        # Why linear1 out_feature is article2*2 rather than linear2 is accusations? Please Think.
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(hidden_size, articles * 2)
        self.linear2 = nn.Linear(hidden_size, accusations)

    def forward(self, inputs):
        output = self.bert(**inputs).last_hidden_state
        last_embedding = output[:, 0, :]
        batch_size = last_embedding.shape[0]
        article_output = self.linear1(self.dropout(last_embedding))  # [batch_size,article_num*2]
        article_output = article_output.view(batch_size, -1, 2)  # [batch_size,article_num,2]
        accusation_output = self.linear2(self.dropout(last_embedding))  # [batch_size,accusation_num]

        return article_output, accusation_output
