import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModel, XLMRobertaModel, RobertaModel

class ClaimVerification(nn.Module):
    def __init__(self, n_classes, 
                 name_model="MoritzLaurer/ernie-m-large-mnli-xnli", 
                 num_experts=4,
                 n_layers=2,
                 dropout_prob=0.3,
                 rank=8,
                 num_experts_per_token=2,
                 temperature=1.0):
        super(ClaimVerification, self).__init__()
        if 'infoxlm' in name_model or 'xlm-roberta' in name_model:
            self.bert = XLMRobertaModel.from_pretrained(name_model) 
        else:
            self.bert = AutoModel.from_pretrained(name_model)
        # self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )
        # x = self.norm(output)
        x = self.drop(output)
        x = self.fc(x)
        return x

# class Adapter(nn.Module):
#     def __init__(self, dim=512, rank=8):
#         super(Adapter, self).__init__()
#         self.adapter_down = nn.Linear(dim, rank)
#         self.adapter_mid = nn.Linear(rank, rank)
#         self.adapter_up = nn.Linear(rank, dim)
#         self.drop = nn.Dropout(0.1)

#     def forward(self, x):
#         down = self.adapter_down(x)           
#         mid = self.adapter_mid(down)           
#         mid = F.relu(mid)                       
#         mid = self.drop(mid)
#         up = self.adapter_up(mid)               
#         return up

# class MoA(nn.Module):
#     def __init__(self, num_experts=8, dim=512, rank=8, num_experts_per_token=2, temperature=1.0):
#         super(MoA, self).__init__()
#         self.num_experts = num_experts
#         self.num_experts_per_token = num_experts_per_token
#         self.temperature = temperature  
#         self.experts = nn.ModuleList([Adapter(dim=dim, rank=rank) for _ in range(num_experts)])

#         self.gate = nn.Linear(dim, num_experts, bias=False)
#         self.layer_norm = nn.LayerNorm(dim, eps=1e-5)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, hidden_states):
#         gate_logits = self.gate(hidden_states)  
#         gate_outputs = F.softmax(gate_logits / self.temperature, dim=-1)

#         topk_gates, topk_indices = torch.topk(gate_outputs, self.num_experts_per_token, dim=-1)
#         topk_gates = topk_gates / torch.sum(topk_gates, dim=-1, keepdim=True).to(hidden_states.dtype)
#         expert_outputs = torch.zeros_like(hidden_states)

#         for i, expert in enumerate(self.experts): 
#             batch_idx, token_idx, topk_idx = torch.where(topk_indices == i)
#             if batch_idx.numel() > 0:
#                 selected_tokens = hidden_states[batch_idx, token_idx]  
#                 expert_out = expert(selected_tokens)  
#                 expert_outputs[batch_idx, token_idx] += (
#                     topk_gates[batch_idx, token_idx, topk_idx].unsqueeze(-1) * expert_out
#                 )

#         output = self.dropout(expert_outputs)
#         output = self.layer_norm(output)
#         return output

# class EncoderLayer(nn.Module):
#     def __init__(self, num_experts, hidden_size, rank, num_experts_per_token):
#         super(EncoderLayer, self).__init__()
#         self.norm = nn.LayerNorm(hidden_size, eps=1e-5)
#         self.dropout = nn.Dropout(0.1)

#         self.adapter = MoA(num_experts=num_experts, dim=hidden_size, rank=rank, num_experts_per_token=num_experts_per_token)

#         self.activation = nn.ReLU()
#         self.linear1 = nn.Linear(hidden_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, hidden_size)

#     def forward(self, text):
#         text_norm = self.norm(text)
#         ff_out = self.linear2(text_norm)
#         ff_out = self.activation(ff_out)
#         ff_out = self.dropout(ff_out)
#         ff_out = self.linear1(ff_out)

#         adapter_out = self.adapter(text_norm)
#         text_embeds = text + ff_out + adapter_out
#         return text_embeds

# class Encoder(nn.Module):
#     def __init__(self, num_experts, hidden_size, rank, num_experts_per_token, n_layers=2):
#         super(Encoder, self).__init__()
#         self.encoders = nn.ModuleList([
#             EncoderLayer(num_experts, hidden_size, rank, num_experts_per_token) for _ in range(n_layers)
#         ])

#     def forward(self, text):
#         for layer in self.encoders:
#             text = layer(text)
#         return text

# class ClaimVerification(nn.Module):
#     def __init__(self, 
#                  n_classes, 
#                  name_model="MoritzLaurer/ernie-m-large-mnli-xnli",
#                  num_experts=4,
#                  n_layers=2,
#                  dropout_prob=0.3,
#                  rank=8,
#                  num_experts_per_token=2,
#                  temperature=1.0):
#         super(ClaimVerification, self).__init__()
        
#         # Load backbone
#         self.bert = XLMRobertaModel.from_pretrained(name_model) if "xlm-roberta" in name_model else AutoModel.from_pretrained(name_model)
#         hidden_dim = self.bert.config.hidden_size

#         # Encoder has MoA
#         self.encoder = Encoder(num_experts, hidden_dim, rank, num_experts_per_token, n_layers)

#         # Max Pooling 
#         self.fc_combine = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.fc_final = nn.Linear(hidden_dim, n_classes)
#         self.drop = nn.Dropout(p=dropout_prob)
#         self.temperature = temperature

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         last_hidden_state = outputs["last_hidden_state"]  

#         encoded_output = self.encoder(last_hidden_state)  

#         cls_token = encoded_output[:, 0, :]  

#         # Max Pooling
#         max_pooled = torch.max(encoded_output, dim=1)[0]  

#         combined_embedding = torch.cat([cls_token, max_pooled], dim=-1)
#         combined_embedding = self.fc_combine(combined_embedding)
#         combined_embedding = self.drop(combined_embedding)

#         logits = self.fc_final(combined_embedding)

#         return logits