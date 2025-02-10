import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# QA
def compute_rationale_regularization(pt, mask=None, lambda_sparse=0.01, lambda_continuity=0.01):
    """
    pt: (batch_size, seq_len, 1) - xác suất rationale mỗi token
    mask: (batch_size, seq_len) - optional, nếu có padding
    lambda_sparse, lambda_continuity: hệ số regularization
    
    Return:
      total_reg_loss: giá trị float (tensor)
    """
    # Bỏ dimension 1 thừa => (batch_size, seq_len)
    pt = pt.squeeze(-1)
    
    if mask is not None:
        pt = pt * mask  # zero out padding positions
    
    # ----- 1) Sparsity: sum of p_i -----
    #    Ở đây dùng L1 sum(pt). 
    #    Hoặc bạn có thể dùng torch.mean(pt) => phạt trung bình p_i.
    sparsity_loss = pt.sum(dim=-1).mean()  # trung bình theo batch
    
    # ----- 2) Continuity: sum (p_i - p_{i+1})^2 -----
    #    Tính cho i từ 0..seq_len-2
    continuity_loss = (pt[:, 1:] - pt[:, :-1])**2
    continuity_loss = continuity_loss.mean()  # trung bình all positions & batch
    
    total_reg_loss = lambda_sparse * sparsity_loss + lambda_continuity * continuity_loss
    
    return total_reg_loss

class RTLoss(nn.Module):
    
    def __init__(self, device = 'cuda'):
        super(RTLoss, self).__init__()
        self.device = device
    
    def forward(self, pt: torch.Tensor, Tagging:  torch.Tensor):
        '''
        Tagging: list paragraphs contain value token. If token of the paragraphas is rationale will labeled 1 and other will be labeled 0 
        
        RT: 
                    p^r_t = sigmoid(w_2*RELU(W_1.h_t))
            
            With:
                    p^r_t constant
                    w_2 (d x 1)
                    W_1 (d x d)
                    h_t (1 x d)
                    
            This formular is compute to each token in paraphase. I has convert into each paraphase
            
                    p^r_t = sigmoid(w_2*RELU(W_1.h))
                    
                    With:
                            p^r (1 x n) with is number of paraphase
                            w_2 (d x 1)
                            W_1 (d x d)
                            h (n x d) 
                            
        '''
        
        Tagging = torch.tensor(Tagging, dtype=torch.float32).to(pt.device)
                
        total_loss = torch.tensor(0, dtype= torch.float32).to(pt.device)
        
        N = pt.shape[0]
                
        for i, text in enumerate(pt):
            T = len(Tagging[i])
            Lrti = -(1/T) * (Tagging[i]@torch.log(text) + (1.0 - Tagging[i]) @ torch.log(1.0 - text) )[0]
            total_loss += Lrti
            
        return total_loss/N


class comboLoss(nn.Module):
    def __init__(self, config):
        
        super(comboLoss, self).__init__()
        self.alpha = config.alpha
        self.beta = config.beta
        # self.BaseLoss = BaseLoss()
        self.RTLoss = RTLoss()
        self.config = config
        
    def forward(self, output: dict):
        attention_mask = output['attention_mask']
        start_logits = output['start_logits']
        end_logits = output['end_logits']
        
        start_positions = output['start_positions']
        end_positions = output['end_positions']
        
        Tagging = output['Tagging']
        pt = output['pt']

        loss_base = 0
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss_base = (start_loss + end_loss) / 2
        retation_tagg_loss  = self.RTLoss(pt = pt, Tagging = Tagging)
        # retation_tagg_loss = nn.BCELoss()(pt, Tagging)
        # retation_tagg_loss = 0
        crr_loss = compute_rationale_regularization(pt, attention_mask)
        total_loss = self.alpha*loss_base+ crr_loss + self.beta*retation_tagg_loss
        
        return total_loss, loss_base, crr_loss, retation_tagg_loss
    

# Classify
class focal_loss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """

#         if not torch.jit.is_scripting() and not torch.jit.is_tracing():
#             _log_api_usage_once(sigmoid_focal_loss)
        p = torch.sigmoid(inputs) 
        # targets = torch.argmax(targets, dim=1)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
    
class AdaptiveFocalMarginLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, alpha=0.5, margin=0.2, reduction='mean'):
        """
        Hàm loss kết hợp Focal Loss và Margin Loss.
        
        Args:
        - class_weights: Tensor chứa trọng số cho từng class (shape: [num_classes]).
        - gamma: Tham số Focal Loss.
        - alpha: Hệ số điều chỉnh tác động của Margin Loss.
        - margin: Giá trị biên độ cho Margin Loss.
        - reduction: 'mean', 'sum', hoặc 'none' để giảm kích thước loss.
        """
        super(AdaptiveFocalMarginLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
        - logits: Đầu ra chưa qua softmax từ mô hình (shape: [batch_size, num_classes]).
        - targets: Ground truth labels (shape: [batch_size]).
        
        Returns:
        - Loss (scalar hoặc tensor tùy thuộc vào reduction).
        """
        num_classes = logits.size(-1)
        probs = F.softmax(logits, dim=-1) 
        
        pt = (probs * targets).sum(dim=-1)
        
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = -focal_weight * torch.log(pt + 1e-8)
        
        margin_loss = torch.clamp(self.margin - pt, min=0) ** 2
        
        loss = focal_loss + self.alpha * margin_loss
        
        if self.class_weights is not None:
            weights = self.class_weights[targets]  
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss