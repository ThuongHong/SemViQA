from transformers import AutoModel, RobertaModel, XLMRobertaModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from models.SMOE.moe import MoELayer

class Rational_Tagging(nn.Module):
    def __init__(self, hidden_size):
        super(Rational_Tagging, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, 1)

    def forward(self, h_t):
        h_1 = self.W1(h_t)
        h_1 = F.relu(h_1) 
        p = self.w2(h_1)
        p = torch.sigmoid(p)  # (batch_size, seq_len, 1) 
        return p

class QATC(nn.Module):
    def __init__(self, config):
        super(QATC, self).__init__()
        
        self.config = config
        if "deberta" in self.config.model_name:
            self.model = AutoModel.from_pretrained(self.config.model_name)
        elif "jina" in self.config.model_name:
            self.model = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True, torch_dtype=torch.float32)
        elif "info" in self.config.model_name:
            self.model = XLMRobertaModel.from_pretrained(self.config.model_name)
        else:
            self.model = RobertaModel.from_pretrained(self.config.model_name)

        # Freeze the text encoder
        if self.config.freeze_text_encoder:
            print("Apply freeze model")
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(0.1)

        # Tầng đầu ra sẽ dự đoán `start` và `length` (thay vì `start` và `end`)
        self.qa_outputs = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.model.config.hidden_size, 2),  # (start, length)
        )
        
        self.tagging = Rational_Tagging(self.model.config.hidden_size)
        self.model.pooler = None

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        qa_outputs = output[0]

        # Rational tagging
        pt = self.tagging(qa_outputs)
        
        # Dự đoán start_logits và length_logits
        logits = self.qa_outputs(qa_outputs)
        start_logits, length_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        length_logits = length_logits.squeeze(-1).contiguous()

        # Chuyển length logits thành giá trị nguyên dương
        length = torch.clamp(torch.round(F.relu(length_logits)), min=0, max=input_ids.shape[1] - 1)

        # Tính end_logits = start_logits + length
        end_logits = start_logits + length

        return pt, start_logits, end_logits
