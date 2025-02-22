import torch.nn as nn
from transformers import AutoModel, XLMRobertaModel

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1)  # Tính attention weights
        pooled_output = torch.sum(attn_weights * x, dim=1)  # Weighted sum của tất cả token
        return pooled_output

class ClaimVerification(nn.Module):
    def __init__(self, name_model):
        super(ClaimVerification, self).__init__()
        self.bert = AutoModel.from_pretrained(name_model)
        self.attn_pooling = AttentionPooling(self.bert.config.hidden_size)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = self.attn_pooling(outputs.last_hidden_state)
        logits = self.fc(self.drop(pooled_output))
        return logits

