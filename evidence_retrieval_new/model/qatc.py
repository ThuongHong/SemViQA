from transformers import AutoModel, RobertaModel, XLMRobertaModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CosineSimilarity

class SentenceRelevanceScorer(nn.Module):
    """Module tính điểm liên quan của câu với claim."""
    def __init__(self, hidden_size):
        super(SentenceRelevanceScorer, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)  # Dự đoán điểm liên quan

    def forward(self, sentence_embeddings):
        scores = self.fc(sentence_embeddings).squeeze(-1)  # Shape: (batch_size, num_sentences)
        return torch.sigmoid(scores)  # Chuẩn hóa về [0, 1]

class RationalTagging(nn.Module):
    """Tagging token đóng góp vào câu trả lời."""
    def __init__(self, hidden_size):
        super(RationalTagging, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        x = F.relu(self.fc1(hidden_states))
        return torch.sigmoid(self.fc2(x))  # Shape: (batch_size, seq_length)

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

        if self.config.freeze_text_encoder:
            print("Applying freeze to the model.")
            for param in self.model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.1)

        # Mô-đun đánh giá độ liên quan của câu
        self.sentence_scorer = SentenceRelevanceScorer(self.model.config.hidden_size)

        # Mô-đun đánh dấu token liên quan
        self.tagging = RationalTagging(self.model.config.hidden_size)

        # Mô-đun tìm vị trí bắt đầu và kết thúc
        self.qa_outputs = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.model.config.hidden_size, 2),
        )

        # Cosine similarity giữa claim và subtexts
        self.cosine_similarity = CosineSimilarity(dim=-1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # === 1. TÁCH CLAIM & SUBTEXTS ===
        sep_token_id = 2  # `[SEP]` token ID (có thể thay đổi dựa trên tokenizer)
        batch_size, seq_length, hidden_size = sequence_output.shape

        claim_embeds = []
        subtext_embeds = []
        for i in range(batch_size):
            sep_indices = (input_ids[i] == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_indices) == 0:
                claim_embeds.append(sequence_output[i, 0].unsqueeze(0))  # Lấy token đầu làm claim
                subtext_embeds.append(sequence_output[i, 1:].unsqueeze(0))  # Phần còn lại là subtexts
            else:
                claim_embeds.append(sequence_output[i, 0:sep_indices[0]].mean(dim=0, keepdim=True))
                subtext_embeds.append(sequence_output[i, sep_indices[0] + 1:].unsqueeze(0))

        claim_embeds = torch.cat(claim_embeds, dim=0)  # (batch_size, hidden_size)
        subtext_embeds = torch.cat(subtext_embeds, dim=0)  # (batch_size, num_sentences, hidden_size)

        # === 2. TÍNH ĐỘ LIÊN QUAN CỦA CÁC CÂU ===
        sentence_scores = self.sentence_scorer(subtext_embeds)  # (batch_size, num_sentences)
        attention_weights = torch.softmax(sentence_scores, dim=-1)  # Chuẩn hóa

        # === 3. TAGGING TOKEN TRONG CÁC CÂU LIÊN QUAN ===
        token_weights = self.tagging(sequence_output)  # (batch_size, seq_length)

        # Kết hợp trọng số câu và token để tạo attention tổng hợp
        combined_attention = attention_weights.unsqueeze(-1) * token_weights  # (batch_size, seq_length)

        # === 4. DỰ ĐOÁN START/END VỚI TRỌNG SỐ TỔNG HỢP ===
        logits = self.qa_outputs(sequence_output)  # (batch_size, seq_length, 2)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1) * combined_attention  # Điều chỉnh start logits
        end_logits = end_logits.squeeze(-1) * combined_attention  # Điều chỉnh end logits

        return combined_attention, start_logits, end_logits
