import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticPromptBank(nn.Module):
    def __init__(self, num_classes, dim=64, init_center=None):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        
        if init_center is not None:
            pc = torch.tensor(init_center, dtype=torch.float32)
            self.prompts = nn.Parameter(pc.clone()) # [K, dim]
        else:
            self.prompts = nn.Parameter(torch.randn(num_classes, dim) * 0.01)

    def forward(self):
        return self.prompts

    def get_prompt_by_label(self, labels):
        return self.prompts[labels]

    def soft_mix(self, feat, topk=2, temp=0.07):
        prompts = F.normalize(self.prompts, dim=1)  # [K, dim]
        query = F.normalize(feat, dim=1)            # [B, dim]
        
        sims = torch.matmul(query, prompts.t())     # [B, K]
        probs = F.softmax(sims / temp, dim=1)       # [B, K]

        if topk is not None and topk < self.num_classes:
            _, topk_idx = torch.topk(probs, topk, dim=1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, topk_idx, 1.0)
            probs = probs * mask
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-9)

        mixed_prompt = torch.matmul(probs, self.prompts) # [B, dim]
        return mixed_prompt, probs

class PC_FiLM(nn.Module):
    def __init__(self, dim=64, out_channel=32, hidden=128, scale_init=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU()
        )
        self.gamma = nn.Linear(hidden, out_channel)
        self.beta = nn.Linear(hidden, out_channel)

        nn.init.constant_(self.gamma.bias, 0.0)
        nn.init.constant_(self.beta.bias, 0.0)
        for m in [self.gamma, self.beta]:
            nn.init.normal_(m.weight, mean=0.0, std=scale_init)

    def forward(self, cond):
        h = self.mlp(cond)
        return self.gamma(h), self.beta(h)

    def apply_film(self, feat, cond):
        """
        feat: [B, C, H, W]
        cond: [B, dim]
        """
        gamma, beta = self.forward(cond)
        gamma = gamma.view(-1, gamma.size(1), 1, 1)
        beta = beta.view(-1, beta.size(1), 1, 1)
        return feat * (1.0 + gamma) + beta

class PGCA(nn.Module):
    def __init__(self, channel=32, dim=64, num_queries=4, num_heads=2, proj_hidden=64):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.scale = (channel // num_heads) ** -0.5
        
        self.query_proj = nn.Sequential(
            nn.Linear(dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, num_queries * channel)
        )
        self.kv_proj = nn.Linear(channel, channel * 2)
        self.out_proj = nn.Linear(num_queries * channel, channel)

    def forward(self, feat, cond):
        B, C, H, W = feat.shape
        tokens = feat.view(B, C, -1).permute(0, 2, 1) # [B, N, C]

        q = self.query_proj(cond).view(B, self.num_queries, C) # [B, Q, C]
        kv = self.kv_proj(tokens).view(B, -1, 2, C)            # [B, N, 2, C]
        
        k, v = kv[:, :, 0, :], kv[:, :, 1, :]

        head_dim = C // self.num_heads
        Q = q.view(B, self.num_queries, self.num_heads, head_dim).permute(0, 2, 1, 3)
        K = k.view(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
        V = v.view(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, self.num_queries, C)
        
        delta = self.out_proj(out.view(B, -1)) # [B, C]
        delta = torch.tanh(delta).view(B, C, 1, 1)
        
        return feat * (1.0 + delta)

class PromptGateGen(nn.Module):
    def __init__(self, dim=64, channel=32, per_channel=False):
        super().__init__()
        self.per_channel = per_channel
        out_features = channel if per_channel else 1
        
        self.fc = nn.Linear(dim, out_features)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, cond):
        alpha = torch.sigmoid(self.fc(cond))
        if self.per_channel:
            return alpha.view(-1, alpha.size(1), 1, 1) # [B, C, 1, 1]
        return alpha.view(-1, 1, 1, 1)                 # [B, 1, 1, 1]