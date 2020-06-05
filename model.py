import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from layers import * 

class RotAtte(nn.Module):
    def __init__(self, n_ent, n_rel, in_dim, out_dim, n_heads=1, input_drop=0.5, negative_rate = 10, margin=6.0, epsilon=2.0, batch_size=None, device="cuda"):
        super().__init__()

        self.n_ent = n_ent 
        self.n_rel = n_rel 
        self.in_dim = in_dim 
        self.out_dim = out_dim
        self.n_heads = n_heads

        self.device = device

        self.a = nn.ModuleList([
            KGLayer(
                n_ent, n_rel, in_dim, out_dim, input_drop, margin=margin, epsilon=epsilon
            )
        for _ in range(self.n_heads)])

        self.rotate = RotAttLayer(n_ent, n_rel, in_dim, out_dim, n_heads=1, input_drop=0.5, negative_rate = negative_rate, margin=margin, epsilon=epsilon, batch_size=batch_size, device=device) 

        self.ent_transform = nn.Linear(n_heads * out_dim, out_dim).to(device)
        self.rel_transform = nn.Linear(n_heads * out_dim, out_dim // 2).to(device)
    
    def forward(self, sample, triplets, mode="single"):
        out = [a(triplets) for a in self.a]
        ent_embed = self.ent_transform(torch.cat([o[0] for o in out], dim=1))
        rel_embed = self.rel_transform(torch.cat([o[1] for o in out], dim=1))

        if mode == 'single':
            mask_indices = torch.unique(torch.cat([ sample[:, 0], sample[:, 2] ]))
        elif mode == 'tail-batch':
            mask_indices = torch.unique(torch.cat([ sample[0][:, 0], sample[0][:, 2], sample[1].flatten()]))
        elif mode == 'head-batch':
            mask_indices = torch.unique(torch.cat([ sample[1][:, 0], sample[1][:, 2], sample[0].flatten()]))
        mask = torch.zeros(self.n_ent).to(self.device)
        mask[mask_indices] = 1.0
        ent_embed = mask.unsqueeze(-1).expand_as(ent_embed) * ent_embed 
        score = self.rotate(sample, ent_embed, rel_embed, mode)

        return score 
    
    def regularization(self):
        pass 