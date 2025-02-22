from transformers import AutoModel, RobertaModel, XLMRobertaModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from model.SMOE.moe import MoELayer
 
class Rational_Tagging(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(Rational_Tagging, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, h_t):
        h_t = h_t.permute(1, 0, 2)   
        attn_output, _ = self.attn(h_t, h_t, h_t)  
        gated_h = torch.sigmoid(self.gate(h_t)) * attn_output  
        h_out = self.norm(gated_h + h_t)  
        h_out = self.dropout(h_out)
        h_out = h_out.permute(1, 0, 2)   
        p = torch.sigmoid(self.fc(h_out))  
        return p
    
class TextEncoder(nn.Module):
    def __init__(self, dim):
        super(TextEncoder, self).__init__()
        self.global_proj = nn.Linear(dim, dim)
        self.local_proj = nn.Linear(dim, dim)
        self.qa_proj = nn.Linear(dim, dim)

    def forward(self, hidden_states):        
        # Tạo 3 dạng representation khác nhau
        global_view = self.global_proj(hidden_states)  # Toàn bộ văn bản
        local_view = self.local_proj(hidden_states[:, :256, :])  # Phần đầu tiên
        qa_view = self.qa_proj(hidden_states)  # Thông tin tập trung vào câu hỏi
        
        return global_view, local_view, qa_view


class GatedAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GatedAttention, self).__init__()
        self.gate = nn.Linear(hidden_size * 3, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, global_view, local_view, qa_view):
        combined = torch.cat([global_view, local_view, qa_view], dim=-1)
        gate_values = self.softmax(self.gate(combined))
        
        fused_representation = gate_values[:, 0:1] * global_view + \
                               gate_values[:, 1:2] * local_view + \
                               gate_values[:, 2:3] * qa_view
        return fused_representation


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
        self.qa_outputs = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.model.config.hidden_size, 2),
        )
        self.encoder = TextEncoder(self.model.config.hidden_size)
        self.gated_attention = GatedAttention(self.model.config.hidden_size)
        self.tagging = Rational_Tagging(self.model.config.hidden_size) 
        self.model.pooler = None
    def forward(self, input_ids, attention_mask):
        output = self.model( input_ids= input_ids, attention_mask = attention_mask)
        qa_ouputs = output[0]

        pt = self.tagging(qa_ouputs)

        global_view, local_view, qa_view = self.encoder(qa_ouputs)
        fused_representation = self.gated_attention(global_view, local_view, qa_view)

        logits = self.qa_outputs(fused_representation) 
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        return pt, start_logits, end_logits
