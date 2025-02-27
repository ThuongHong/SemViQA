# from transformers import AutoModel, RobertaModel, XLMRobertaModel
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import copy
# from model.SMOE.moe import MoELayer

# class Rational_Tagging(nn.Module):
#     def __init__(self, hidden_size):
#         super(Rational_Tagging, self).__init__()
#         self.W1 = nn.Linear(hidden_size, hidden_size)
#         self.w2 = nn.Linear(hidden_size, 1)

#     def forward(self, h_t):
#         h_1 = self.W1(h_t)
#         h_1 = F.relu(h_1) 
#         p = self.w2(h_1)
#         p = torch.sigmoid(p) # (batch_size, seq_len, 1) 
#         return p

# class HierarchicalAttention(nn.Module):
#     """Multi-level attention mechanism for contextual understanding of evidence spans"""
#     def __init__(self, hidden_size, num_heads=8):
#         super(HierarchicalAttention, self).__init__()
#         self.token_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
#         self.span_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(hidden_size)
#         self.norm2 = nn.LayerNorm(hidden_size)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 4),
#             nn.GELU(),
#             nn.Linear(hidden_size * 4, hidden_size)
#         )
        
#     def forward(self, hidden_states, attention_mask):
#         # Convert attention mask to correct format for multihead attention
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
#         # Token-level attention
#         token_context, _ = self.token_attention(
#             hidden_states, hidden_states, hidden_states, 
#             key_padding_mask=(1 - attention_mask).bool()
#         )
#         hidden_states = self.norm1(hidden_states + token_context)
        
#         # Span-level attention with sliding window approach
#         span_context, _ = self.span_attention(
#             hidden_states, hidden_states, hidden_states,
#             key_padding_mask=(1 - attention_mask).bool()
#         )
        
#         hidden_states = self.norm2(hidden_states + span_context)
#         hidden_states = hidden_states + self.ffn(hidden_states)
        
#         return hidden_states
 

# class EvidenceAwareGating(nn.Module):
#     """Gates information flow based on token's evidence relevance"""
#     def __init__(self, hidden_size):
#         super(EvidenceAwareGating, self).__init__()
#         self.evidence_query = nn.Parameter(torch.randn(hidden_size))
#         self.gate_proj = nn.Linear(hidden_size, 1)
#         self.transform = nn.Linear(hidden_size, hidden_size)
        
#     def forward(self, hidden_states, rationale_probs):
#         # Calculate evidence relevance scores
#         evidence_scores = self.gate_proj(hidden_states).squeeze(-1)
        
#         # Combine with token-level rationale probabilities
#         combined_scores = evidence_scores * rationale_probs.squeeze(-1)
#         gate_values = torch.sigmoid(combined_scores).unsqueeze(-1)
        
#         # Apply gating mechanism
#         transformed = self.transform(hidden_states)
#         gated_output = hidden_states + gate_values * transformed
        
#         return gated_output

# class QATC(nn.Module):
#     def __init__(self, config):
#         super(QATC, self).__init__()
        
#         self.config = config
#         # Load pre-trained language model
#         if "deberta" in self.config.model_name:
#             self.model = AutoModel.from_pretrained(self.config.model_name)
#         elif "jina" in self.config.model_name:
#             self.model = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True, torch_dtype=torch.float32)
#         elif "info" in self.config.model_name:
#             self.model = XLMRobertaModel.from_pretrained(self.config.model_name)
#         else:
#             self.model = RobertaModel.from_pretrained(self.config.model_name)
            
#         # Freeze the text encoder if specified
#         if self.config.freeze_text_encoder:
#             print("Apply freeze model")
#             for param in self.model.parameters():
#                 param.requires_grad = False
        
#         hidden_size = self.model.config.hidden_size
        
#         # Rational tagging and evidence-aware modules
#         self.tagging = Rational_Tagging(hidden_size)
#         self.evidence_gating = EvidenceAwareGating(hidden_size)
        
#         # Multi-level hierarchical attention
#         self.hierarchical_attention = HierarchicalAttention(hidden_size)
        
#         # Dropout for regularization
#         self.dropout = nn.Dropout(0.1)
        
#         # Global context integration
#         self.global_context = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.GELU()
#         )
        
#         # Uncertainty modeling for start and end prediction
#         self.uncertainty_modeling = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, 2)  # uncertainty for start and end
#         )
        
#         # Span scoring with adaptive thresholding
#         self.span_scorer = nn.Linear(hidden_size * 2, 1)
        
#         # Final boundary prediction layers
#         self.start_predictor = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.GELU(),
#             self.dropout,
#             nn.Linear(hidden_size, 1)
#         )
        
#         self.end_predictor = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.LayerNorm(hidden_size),
#             nn.GELU(),
#             self.dropout,
#             nn.Linear(hidden_size, 1)
#         )
        
#         # Remove pooler to focus on token-level representations
#         self.model.pooler = None
    
    
#     def forward(self, input_ids, attention_mask):
#         # Get base embeddings from language model
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs[0]
        
#         # 1. Rational tagging to identify potential evidence tokens
#         rationale_probs = self.tagging(hidden_states)
        
#         # 2. Apply evidence-aware gating to focus on relevant tokens
#         gated_hidden_states = self.evidence_gating(hidden_states, rationale_probs)
        
#         # 3. Apply hierarchical attention for contextual understanding
#         contextual_states = self.hierarchical_attention(gated_hidden_states, attention_mask)
        
#         # 4. Global context integration
#         batch_size, seq_len, hidden_dim = contextual_states.shape
#         global_context = torch.mean(contextual_states * rationale_probs, dim=1, keepdim=True)
#         global_context = self.global_context(global_context)
#         enhanced_states = contextual_states + global_context.expand(-1, seq_len, -1)
            
#         # 7. Predict uncertainty scores
#         uncertainty_scores = self.uncertainty_modeling(enhanced_states)
        
#         # 8. Augment representation with global context for final prediction
#         augmented_states = torch.cat([enhanced_states, 
#                                      global_context.expand(-1, seq_len, -1)], dim=-1)
        
#         # 9. Final boundary prediction
#         start_logits = self.start_predictor(augmented_states).squeeze(-1)
#         end_logits = self.end_predictor(augmented_states).squeeze(-1)
        
#         # Adjust logits based on uncertainty
#         start_uncertainty, end_uncertainty = uncertainty_scores.split(1, dim=-1)
#         start_logits = start_logits * (1 - torch.sigmoid(start_uncertainty.squeeze(-1)))
#         end_logits = end_logits * (1 - torch.sigmoid(end_uncertainty.squeeze(-1)))
        
#         return rationale_probs, start_logits, end_logits



from transformers import AutoModel, RobertaModel, XLMRobertaModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy

class Rational_Tagging(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(Rational_Tagging, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size // 2)
        self.w3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, h_t):
        h_norm = self.layer_norm(h_t)
        h_1 = self.W1(h_norm)
        h_1 = F.gelu(h_1)  # Using GELU instead of ReLU for better gradient flow
        h_1 = self.dropout(h_1)
        h_2 = self.W2(h_1)
        h_2 = F.gelu(h_2)
        h_2 = self.dropout(h_2)
        p = self.w3(h_2)
        p = torch.sigmoid(p)  # (batch_size, seq_len, 1)
        return p

class HierarchicalAttention(nn.Module):
    """Enhanced multi-level attention mechanism with residual connections"""
    def __init__(self, hidden_size, num_heads=8, dropout_rate=0.1):
        super(HierarchicalAttention, self).__init__()
        self.token_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout_rate)
        self.span_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, hidden_states, attention_mask):
        # Token-level attention with proper masking
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        token_context, _ = self.token_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=(1 - attention_mask).bool()
        )
        hidden_states = residual + self.dropout(token_context)
        
        # Span-level attention
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        span_context, _ = self.span_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=(1 - attention_mask).bool()
        )
        hidden_states = residual + self.dropout(span_context)
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states

class EvidenceAwareGating(nn.Module):
    """Enhanced gating with better information flow"""
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(EvidenceAwareGating, self).__init__()
        self.evidence_transform = nn.Linear(hidden_size, hidden_size)
        self.gate_proj = nn.Linear(hidden_size, 1)
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, hidden_states, rationale_probs):
        # Calculate evidence relevance with normalization
        norm_states = self.layer_norm1(hidden_states)
        evidence_transform = self.evidence_transform(norm_states)
        evidence_scores = self.gate_proj(F.gelu(evidence_transform)).squeeze(-1)
        
        # Combine with token-level rationale probabilities using scaled sigmoid
        combined_scores = evidence_scores * rationale_probs.squeeze(-1)
        gate_values = torch.sigmoid(combined_scores).unsqueeze(-1)
        
        # Apply gating mechanism with residual connection
        norm_states2 = self.layer_norm2(hidden_states)
        transformed = self.transform(norm_states2)
        transformed = self.dropout(transformed)
        gated_output = hidden_states + gate_values * transformed
        
        return gated_output

class QATC(nn.Module):
    def __init__(self, config):
        super(QATC, self).__init__()
        
        self.config = config
        # Load pre-trained language model
        if "deberta" in self.config.model_name:
            self.model = AutoModel.from_pretrained(self.config.model_name)
        elif "jina" in self.config.model_name:
            self.model = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True, torch_dtype=torch.float32)
        elif "info" in self.config.model_name:
            self.model = XLMRobertaModel.from_pretrained(self.config.model_name)
        else:
            self.model = RobertaModel.from_pretrained(self.config.model_name)
            
        # Freeze the text encoder if specified
        if self.config.freeze_text_encoder:
            print("Apply freeze model")
            for param in self.model.parameters():
                param.requires_grad = False
        
        hidden_size = self.model.config.hidden_size
        dropout_rate = 0.1
        
        # Improved rational tagging and evidence-aware modules
        self.tagging = Rational_Tagging(hidden_size, dropout_rate)
        self.evidence_gating = EvidenceAwareGating(hidden_size, dropout_rate)
        
        # Enhanced multi-level hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(hidden_size, num_heads=12, dropout_rate=dropout_rate)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Global context integration with enhanced pooling
        self.global_context = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Context-aware boundary detection 
        self.boundary_context = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final boundary prediction layers with deeper networks
        self.start_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.end_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Remove pooler to focus on token-level representations
        self.model.pooler = None
        
        # Cross-attention between start and end representations
        self.cross_attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True, dropout=dropout_rate)
        self.cross_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, input_ids, attention_mask):
        # Get base embeddings from language model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        
        # 1. Improved rational tagging to identify potential evidence tokens
        rationale_probs = self.tagging(hidden_states)
        
        # 2. Apply enhanced evidence-aware gating to focus on relevant tokens
        gated_hidden_states = self.evidence_gating(hidden_states, rationale_probs)
        
        # 3. Apply hierarchical attention for contextual understanding
        contextual_states = self.hierarchical_attention(gated_hidden_states, attention_mask)
        
        # 4. Enhanced global context integration with weighted pooling
        batch_size, seq_len, hidden_dim = contextual_states.shape
        attention_weights = rationale_probs * attention_mask.unsqueeze(-1).float()
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-12)
        global_context = torch.sum(contextual_states * attention_weights, dim=1, keepdim=True)
        global_context = self.global_context(global_context)
        
        # 5. Enhanced representation with global context
        enhanced_states = contextual_states + global_context.expand(-1, seq_len, -1)
        
        # 6. Cross-attention between start and end position modeling
        cross_states = self.cross_norm(enhanced_states)
        cross_attended, _ = self.cross_attention(
            cross_states, cross_states, cross_states,
            key_padding_mask=(1 - attention_mask).bool()
        )
        boundary_states = enhanced_states + self.dropout(cross_attended)
            
        # 7. Augment representation with global context for final prediction
        augmented_states = torch.cat([
            boundary_states, 
            global_context.expand(-1, seq_len, -1)
        ], dim=-1)
        
        # 8. Final boundary prediction with optimized predictors
        start_logits = self.start_predictor(augmented_states).squeeze(-1)
        end_logits = self.end_predictor(augmented_states).squeeze(-1)
        
        # 9. Apply attention mask to ensure padding tokens have large negative scores
        start_logits = start_logits + (1 - attention_mask) * -10000.0
        end_logits = end_logits + (1 - attention_mask) * -10000.0
        
        return rationale_probs, start_logits, end_logits