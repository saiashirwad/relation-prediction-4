import IPython
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import openke.module.model as model
from layers import KGLayer, KGLayer2
Model = model.Model


class RotAtte(Model):
    def __init__(self, n_ent, n_rel, in_dim, out_dim, facts, encoder="rotate", n_heads=1,
                 n_layers=1, input_drop=0.5, negative_rate=10, margin=6.0,
                 epsilon=2.0, batch_size=None, device="cuda", scatter=False,
                 multiplier=1, ent_embed=None, rel_embed=None):
        super(RotAtte, self).__init__(n_ent, n_rel)

        self.n_ent = n_ent
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.device = device
        self.scatter = scatter
        self.multiplier = multiplier
        if ent_embed != None:
            # self.ent_embed = nn.Embedding(n_ent, out_dim).to(device)
            # self.rel_embed = nn.Embedding(n_rel, out_dim).to(device)
            # self.rel_embed = rel_embed
            self.stage = 2
            # self.ent_embed.weight.data = ent_embed
            # self.rel_embed.weight.data = rel_embed
            self.ent_embed = nn.Parameter(ent_embed)
            self.rel_embed = nn.Parameter(rel_embed)
        else:
            self.stage = 1
            self.a = nn.ModuleList([
                KGLayer(
                    n_ent, n_rel, in_dim, out_dim, input_drop, margin=margin, epsilon=epsilon, scatter=scatter
                )
                for _ in range(self.n_heads)])

            if n_layers == 2:
                self.a2 = KGLayer(n_ent, n_rel, self.n_heads * out_dim, out_dim, input_drop,
                                  margin=margin, epsilon=epsilon, concat=False, scatter=scatter)
            if self.n_layers == 1:
                self.ent_transform = nn.Linear(
                    n_heads * out_dim, out_dim).to(device)
                if encoder == "rotate":
                    self.rel_transform = nn.Linear(
                        n_heads * out_dim, out_dim // 2).to(device)
                else:
                    self.rel_transform = nn.Linear(
                        n_heads * out_dim, out_dim
                    ).to(device)
            else:
                self.rel_transform = nn.Linear(
                    out_dim, out_dim // 2).to(device)
            self.ent_embed = None
            self.rel_embed = None

        self.ent_tot = n_ent
        self.rel_tot = n_rel

        self.facts = facts
        self.encoder_fn = encoder
        if encoder == "rotate":
            self.encoder = RotatE(n_ent, n_rel, out_dim, margin, epsilon)
        elif encoder == "complex":
            self.encoder = ComplEx(n_ent, n_rel, out_dim)
    
    def get_embeddings(self):
        return self.ent_embed, self.rel_embed

    def save_embeddings(self):
        self.ent_embed, self.rel_embed = self.run_kgatt()

        print("Saved embeddings")

    def regularization(self, data):
        return self.encoder.regularization(data, self.ent_embed, self.rel_embed)

    def run_kgatt(self):
        out = [a(self.facts) for a in self.a]
        if self.n_layers == 2:
            ent_embed = torch.cat([o[0] for o in out], dim=1)
            rel_embed = torch.cat([o[1] for o in out], dim=1)

            out = self.a2(self.facts, ent_embed, rel_embed)

            ent_embed = F.elu(out[0])
            rel_embed = out[1]
            rel_embed = self.rel_transform(rel_embed)
        else:
            if self.n_heads > 1:
                ent_embed = self.ent_transform(
                    torch.cat([o[0] for o in out], dim=1))
            else:
                ent_embed, _ = out[0]
            rel_embed = self.rel_transform(
                torch.cat([o[1] for o in out], dim=1))

        return ent_embed, rel_embed

    def forward(self, data):
        if self.stage == 1:
            ent_embed, rel_embed = self.run_kgatt()
            self.ent_embed = ent_embed
            self.rel_embed = rel_embed

            mask = torch.zeros(self.n_ent).to(self.device)
            mask_indices = torch.cat(
                (data['batch_h'], data['batch_t'])).unique()
            mask[mask_indices] = 1.0
            ent_embed = mask.unsqueeze(-1).expand_as(ent_embed) * ent_embed

            return self.encoder(data, ent_embed, rel_embed)
        if self.stage == 2:
            return self.encoder(data, self.ent_embed, self.rel_embed)

    def predict(self, data):
        score = - self.encoder(data, self.ent_embed,
                               self.rel_embed) * self.multiplier
        return score.cpu().data.numpy()


class RotatE(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, margin=6.0, epsilon=2.0):
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

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

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

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()

    def forward(self, data, ent_embed, rel_embed):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = ent_embed[batch_h]
        t = ent_embed[batch_t]
        r = rel_embed[batch_r]
        score = self.margin - self._calc(h, t, r, mode)
        return score


class ComplEx(Model):
    def __init__(self, ent_tot, rel_tot, dim=100):
        super(ComplEx, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        # self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
        # self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
        # self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)
        # self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)

        # nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        # nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        # nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        # nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )

    def forward(self, data, ent_embed, rel_embed):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        ent_re_embeddings, ent_im_embeddings = torch.chunk(
            ent_embed, 2, dim=-1)
        rel_re_embeddings, rel_im_embeddings = torch.chunk(
            rel_embed, 2, dim=-1)
        h_re = ent_re_embeddings[batch_h]
        h_im = ent_im_embeddings[batch_h]
        t_re = ent_re_embeddings[batch_t]
        t_im = ent_im_embeddings[batch_t]
        r_re = rel_re_embeddings[batch_r]
        r_im = rel_im_embeddings[batch_r]
        # IPython.embed()
        score = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
        # IPython.embed()
        print(score)
        return score

    def regularization(self, data, ent_embed, rel_embed):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        ent_re_embeddings, ent_im_embeddings = torch.chunk(
            ent_embed, 2, dim=-1)
        rel_re_embeddings, rel_im_embeddings = torch.chunk(
            rel_embed, 2, dim=-1)
        h_re = ent_re_embeddings[batch_h]
        h_im = ent_im_embeddings[batch_h]
        t_re = ent_re_embeddings[batch_t]
        t_im = ent_im_embeddings[batch_t]
        r_re = rel_re_embeddings[batch_r]
        r_im = rel_im_embeddings[batch_r]
        regul = (torch.mean(h_re ** 2) +
                 torch.mean(h_im ** 2) +
                 torch.mean(t_re ** 2) +
                 torch.mean(t_im ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6
        # IPython.embed()
        return regul

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()
