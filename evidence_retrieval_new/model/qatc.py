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
        p = torch.sigmoid(p) 
        return p

# class QATC(nn.Module):
#     def __init__(self, config):
#         super(QATC, self).__init__()
        
#         self.config = config
#         if "deberta" in self.config.model_name:
#             self.model = AutoModel.from_pretrained(self.config.model_name)
#         elif "jina" in self.config.model_name:
#             self.model = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True, torch_dtype=torch.float32)
#         elif "info" in self.config.model_name:
#             self.model = XLMRobertaModel.from_pretrained(self.config.model_name)
#         else:
#             self.model = RobertaModel.from_pretrained(self.config.model_name)
#         # Freeze the text encoder
#         if self.config.freeze_text_encoder:
#             print("Apply freeze model")
#             for param in self.model.parameters():
#                 param.requires_grad = False
        
#         # Use FC layer to get the start logits l_start and end logits_end
#         # self.qa_outputs = nn.Linear(self.model.config.hidden_size, 2)
#         self.dropout = nn.Dropout(0.1)
#         self.qa_outputs = nn.Sequential(
#             nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
#             nn.LayerNorm(self.model.config.hidden_size),
#             nn.ReLU(),
#             self.dropout,
#             nn.Linear(self.model.config.hidden_size, 2),
#         )
        
#         self.tagging = Rational_Tagging(self.model.config.hidden_size)
#         self.rationale_fc = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
#         self.model.pooler = None
#     def forward(self, input_ids, attention_mask):
#         output = self.model( input_ids= input_ids, attention_mask = attention_mask)
        
#         qa_ouputs = output[0]
#         pt = self.tagging(qa_ouputs)
        
#         logits = self.qa_outputs(qa_ouputs) 
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()
        
#         return pt, start_logits, end_logits

class EvidenceScorer(nn.Module):
    """Tính điểm liên quan của từng token để xác định câu evidence chính xác nhất."""
    def __init__(self, hidden_size):
        super(EvidenceScorer, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores = self.attn(hidden_states).squeeze(-1)  # Tính trọng số
        scores = scores.masked_fill(attention_mask == 0, -1e9)  # Loại bỏ padding
        scores = torch.softmax(scores, dim=1)  # Chuẩn hóa thành xác suất
        return scores

class AdjustedQAOutputs(nn.Module):
    """Điều chỉnh start-end logits dựa vào điểm của câu evidence tìm được."""
    def __init__(self, hidden_size):
        super(AdjustedQAOutputs, self).__init__()
        self.qa_layer = nn.Linear(hidden_size, 2)

    def forward(self, hidden_states, evidence_scores):
        logits = self.qa_layer(hidden_states)  # Dự đoán start-end logits ban đầu
        start_logits, end_logits = logits.split(1, dim=-1)

        # Điều chỉnh start-end logits dựa trên điểm evidence
        start_logits = start_logits.squeeze(-1) * evidence_scores
        end_logits = end_logits.squeeze(-1) * evidence_scores

        return start_logits, end_logits

class MultiHopReasoning(nn.Module):
    """Mở rộng kiểm tra sang câu evidence thứ 2 nếu câu đầu tiên chưa chính xác."""
    def __init__(self, hidden_size):
        super(MultiHopReasoning, self).__init__()
        self.gnn_layer = nn.Linear(hidden_size, hidden_size)  # Mô phỏng GNN đơn giản
        self.relu = nn.ReLU()

    def forward(self, hidden_states, evidence_scores):
        hop_features = hidden_states * evidence_scores.unsqueeze(-1)
        hop_features = self.gnn_layer(hop_features)
        hop_features = self.relu(hop_features)
        return hop_features

class QATC(nn.Module):
    """Mô hình tổng thể: tìm evidence, điều chỉnh vị trí, và multi-hop reasoning."""
    def __init__(self, config):
        super(QATC, self).__init__()
        
        self.config = config
        self.model = AutoModel.from_pretrained(self.config.model_name)

        # Đóng băng encoder nếu cần
        if self.config.freeze_text_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.1)

        # Module tìm câu evidence
        self.evidence_scorer = EvidenceScorer(self.model.config.hidden_size)

        # Module reasoning nhiều bước
        self.multi_hop_reasoning = MultiHopReasoning(self.model.config.hidden_size)

        # Điều chỉnh start-end dựa vào evidence
        self.adjusted_qa_outputs = AdjustedQAOutputs(self.model.config.hidden_size)
        self.tagging = Rational_Tagging(self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output[0]
        pt = self.tagging(hidden_states)

        # Tính điểm liên quan của từng câu
        evidence_scores = self.evidence_scorer(hidden_states, attention_mask)

        # Thực hiện multi-hop reasoning nếu evidence đầu tiên không chính xác
        enhanced_hidden_states = self.multi_hop_reasoning(hidden_states, evidence_scores)

        # Điều chỉnh start-end logits
        start_logits, end_logits = self.adjusted_qa_outputs(enhanced_hidden_states, evidence_scores)

        return pt, start_logits, end_logits
