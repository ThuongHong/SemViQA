import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange, reduce

class MoELayer(nn.Module):
    def __init__(self, experts, gate, num_experts_per_token, hidden_dim):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts_per_token = num_experts_per_token
        self.num_experts = len(experts)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-05, elementwise_affine=True)
        self.dropout = nn.Dropout(p=0.1, inplace=False)

        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = True

        # self.last_z_loss = None
        self.last_balance_loss = None

    def zloss(self, gate_logits):
        router_z_loss = torch.logsumexp(gate_logits, dim=-1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()
        return router_z_loss

    def balanceloss(self, selected_experts, gate_softmax):
        density_1_proxy = reduce(gate_softmax, 'b t e -> e', 'mean')  # Kích thước: (num_experts,)

        one_hot_gate_indices = F.one_hot(
            rearrange(selected_experts, 'b t k -> b t k'), 
            num_classes=self.num_experts
        ).float()
        
        density_1 = reduce(one_hot_gate_indices, 'b t k e -> e', 'mean')  # Kích thước: (num_experts,)

        balance_loss = (density_1_proxy * density_1).mean() * float(self.num_experts ** 2)
        return balance_loss
    
    def combineloss(self, gate_logits, gate_softmax, selected_experts):
        """
        Tính toán tổng loss của hệ thống.
        """
        balance_loss = self.balanceloss(selected_experts, gate_softmax)
        # z_loss = self.zloss(gate_logits)
        return balance_loss 

    def forward(self, hidden_states, input_tensor):
        gate_logits = self.gate(hidden_states)
        gate_outputs = F.softmax(gate_logits, dim=-1)
        
        topk_gates, topk_indices = torch.topk(gate_outputs, self.num_experts_per_token, dim=-1)
        
        expert_outputs = torch.zeros_like(hidden_states)
        
        for i, expert in enumerate(self.experts): 
            batch_idx, token_idx, topk_idx = torch.where(topk_indices == i) 
            selected_tokens = hidden_states[batch_idx, token_idx]
            expert_output = expert(selected_tokens)   
            expert_outputs[batch_idx, token_idx] += topk_gates[batch_idx, token_idx, topk_idx].unsqueeze(-1) * expert_output
        
        output = self.dropout(expert_outputs)
        output = self.layer_norm(output+input_tensor)

        # self.last_balance_loss = self.combineloss(gate_logits, gate_outputs, topk_indices)
        
        return output
    