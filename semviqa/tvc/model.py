import torch
import torch.nn as nn
from transformers import AutoModel, XLMRobertaModel

class ClaimVerification(nn.Module):
    def __init__(self, n_classes, name_model):
        super(ClaimVerification, self).__init__()
        self.bert = AutoModel.from_pretrained(name_model)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x