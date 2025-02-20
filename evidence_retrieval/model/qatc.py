from transformers import AutoModel, RobertaModel, XLMRobertaModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from model.SMOE.moe import MoELayer
 
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
        self.rationale_fc = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.model.pooler = None
    def forward(self, input_ids, attention_mask):
        output = self.model( input_ids= input_ids, attention_mask = attention_mask)
        
        qa_ouputs = output[0]
        pt = self.tagging(qa_ouputs)

        rationale_embedding = self.rationale_fc(qa_ouputs) * pt
        enhanced_outputs = qa_ouputs + rationale_embedding
        
        logits = self.qa_outputs(enhanced_outputs) 
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        return pt, start_logits, end_logits
