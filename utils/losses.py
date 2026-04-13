import torch
import torch.nn as nn
import torch.nn.functional as F

class SODLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_iou=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.wb = weight_bce
        self.wi = weight_iou

    def forward(self, pred, gt): # pred: [B,1,H,W]
        bce_loss = self.bce(pred, gt)

        pred_prob = torch.sigmoid(pred)
        inter = (pred_prob * gt).view(pred_prob.shape[0], -1).sum(dim=1)
        union = (pred_prob + gt - pred_prob * gt).view(pred_prob.shape[0], -1).sum(dim=1)
        iou = (inter + 1e-9) / (union + 1e-9)
        iou_loss = 1.0 - iou.mean()
        return self.wb * bce_loss + self.wi * iou_loss

class PromptDiversityLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, prompts): # [K, dim]
        p = F.normalize(prompts, dim=1)
        sim = torch.matmul(p, p.t())
        K = p.shape[0]
        loss = 0.0
        count = 0
        for i in range(K):
            for j in range(i+1, K):
                s = sim[i, j]
                if s > self.margin:
                    loss += (s - self.margin)
                    count += 1
        if count == 0:
            return torch.tensor(0., device=prompts.device)
        return loss / count

class CosineContrastLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feats, labels, prompts):
        """
        feats: [B, D] (projected global features)
        labels: [B] int
        prompt_centers: [K, D]
        """
        pos = prompts[labels]  # [B, D]
        pos_sim = F.cosine_similarity(feats, pos).mean()

        # sample negatives (shifted by 1 mod K)
        neg_idx = (labels + 1) % prompts.size(0)
        neg = prompts[neg_idx]
        neg_sim = F.cosine_similarity(feats, neg)
        neg_loss = F.relu(neg_sim - self.margin).mean()
        return (1.0 - pos_sim) + neg_loss
