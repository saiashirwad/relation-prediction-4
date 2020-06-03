import openke.module.model as model 
Model = model.Model

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from layers import * 

import numpy as np 

class RotAtte(Model):
    def __init__(self, n_ent, n_rel, in_dim, out_dim, facts, n_heads=1, input_drop=0.5, 
                negative_rate = 10, margin=6.0, epsilon=2.0, batch_size=None, device="cuda"):
        super(RotAtte, self).__init__(n_ent, n_rel)

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

        self.ent_tot = n_ent 
        self.rel_tot = n_rel

        self.ent_transform = nn.Linear(n_heads * out_dim, out_dim).to(device)
        self.rel_transform = nn.Linear(n_heads * out_dim, out_dim // 2).to(device)

        self.ent_embed = None 
        self.rel_embed = None 

        self.facts = facts

        self.rotate = RotatE(n_ent, n_rel, out_dim, margin, epsilon)

    def save_embeddings(self):
        out = [a(self.facts) for a in self.a]
        self.ent_embed = self.ent_transform(torch.cat([o[0] for o in out], dim=1))
        self.rel_embed = self.rel_transform(torch.cat([o[1] for o in out], dim=1))

        print("Saved embeddings")

    def forward(self, data):
        out = [a(self.facts) for a in self.a]
        ent_embed = self.ent_transform(torch.cat([o[0] for o in out], dim=1))
        rel_embed = self.rel_transform(torch.cat([o[1] for o in out], dim=1))

        mask = torch.zeros(self.n_ent).to(self.device)
        mask_indices = torch.cat((data['batch_h'], data['batch_t'])).unique()
        mask[mask_indices] = 1.0
        ent_embed = mask.unsqueeze(-1).expand_as(ent_embed) * ent_embed 

        return self.rotate(data, ent_embed, rel_embed)

    def predict(self, data):
        score = self.rotate(data, self.ent_embed, self.rel_embed) 
        return score.cpu().data.numpy()



class RotatE(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, margin = 6.0, epsilon = 2.0):
		super(RotatE, self).__init__(ent_tot, rel_tot)

		self.margin = margin
		self.epsilon = epsilon

		self.dim_e = dim 
		self.dim_r = dim // 2

		self.rel_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), 
			requires_grad=False
		)

		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False

	def _calc(self, h, t, r, mode):
		pi = self.pi_const

		re_head, im_head = torch.chunk(h, 2, dim=-1)
		re_tail, im_tail = torch.chunk(t, 2, dim=-1)

		phase_relation = r / (self.rel_embedding_range.item() / pi)

		re_relation = torch.cos(phase_relation)
		im_relation = torch.sin(phase_relation)

		re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
		re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
		im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
		im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
		im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
		re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

		if mode == "head_batch":
			re_score = re_relation * re_tail + im_relation * im_tail
			im_score = re_relation * im_tail - im_relation * re_tail
			re_score = re_score - re_head
			im_score = im_score - im_head
		else:
			re_score = re_head * re_relation - im_head * im_relation
			im_score = re_head * im_relation + im_head * re_relation
			re_score = re_score - re_tail
			im_score = im_score - im_tail

		score = torch.stack([re_score, im_score], dim = 0)
		score = score.norm(dim = 0).sum(dim = -1)
		return score.permute(1, 0).flatten()

	def forward(self, data, ent_embed, rel_embed):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = ent_embed[batch_h]
		t = ent_embed[batch_t]
		r = rel_embed[batch_r]
		score = self.margin - self._calc(h ,t, r, mode)
		return score