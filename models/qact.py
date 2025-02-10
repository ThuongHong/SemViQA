from transformers import AutoModel, RobertaModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from models.SMOE.moe import MoELayer

# Code model baseline 
# class Rational_Tagging(nn.Module):
#     def __init__(self,  hidden_size):
#         super(Rational_Tagging, self).__init__()
#         self.W1 = nn.Linear(hidden_size, hidden_size)
#         self.w2 = nn.Linear(hidden_size, 1)

#     def forward(self, h_t):
#         h_1 = self.W1(h_t)
#         # h_1 = F.relu(h_1)
#         h_1 = F.gelu(h_1)
#         p = self.w2(h_1)
#         # p = torch.sigmoid(p) # (batch_size, seq_len, 1)
#         p = F.softmax(p, dim=1)
#         return p


class Rational_Tagging(nn.Module):
    def __init__(self,  hidden_size):
        super(Rational_Tagging, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, 1)

    def forward(self, h_t):
        h_1 = self.W1(h_t)
        h_1 = F.relu(h_1) 
        p = self.w2(h_1)
        p = torch.sigmoid(p) # (batch_size, seq_len, 1) 
        return p

class QACT(nn.Module):
    def __init__(self, config):
        super(QACT, self).__init__()
        
        self.config = config
        if "deberta" in self.config.model_name:
            self.model = AutoModel.from_pretrained(self.config.model_name)
        elif "jina" in self.config.model_name:
            self.model = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True, torch_dtype=torch.float32)
        else:
            self.model = RobertaModel.from_pretrained(self.config.model_name)
        # Freeze the text encoder
        if self.config.freeze_text_encoder:
            print("Apply freeze model")
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Use FC layer to get the start logits l_start and end logits_end
        # self.qa_outputs = nn.Linear(self.model.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
        self.qa_outputs = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.model.config.hidden_size, 2),
        )
        
        self.tagging = Rational_Tagging(self.model.config.hidden_size)
        if self.config.use_smoe:
            print("Using SMoe")
            # Replace attention output with MoE layer
            for layer in self.model.encoder.layer:
                layer.attention.output = MoELayer(
                    experts=[copy.deepcopy(layer.attention.output.dense) for _ in range(self.config.num_experts)],
                    gate=nn.Sequential(
                        nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.model.config.hidden_size, self.config.num_experts), 
                    ),
                    num_experts_per_token=self.config.num_experts_per_token,
                    hidden_dim = self.model.config.hidden_size
                )
        self.model.pooler = None
    def forward(self, input_ids, attention_mask):
        output = self.model( input_ids= input_ids, attention_mask = attention_mask)
        
        qa_ouputs = output[0]
        pt =  self.tagging(qa_ouputs)
        
        logits = self.qa_outputs(qa_ouputs) 
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        return pt, start_logits, end_logits
