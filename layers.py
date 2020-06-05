import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

import IPython


class SNAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(torch.cuda.is_available()):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None


class SparseNeighborhoodAggregation(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SNAFunction.apply(edge, edge_w, N, E, out_features)


class KGLayer(nn.Module):
    def __init__(self, n_entities, n_relations, in_dim, out_dim, input_drop=0.5,
                 margin=6.0, epsilon=2.0, device="cuda", concat=True):
        super().__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        self.margin = margin
        self.epsilon = epsilon

        self.a = nn.Linear(3 * in_dim, out_dim).to(device)
        nn.init.xavier_normal_(self.a.weight.data, gain=1.414)

        self.a_2 = nn.Linear(out_dim, 1).to(device)
        nn.init.xavier_normal_(self.a_2.weight.data, gain=1.414)

        self.sparse_neighborhood_aggregation = SparseNeighborhoodAggregation()

        self.concat = concat

        if concat:
            self.ent_embed_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.out_dim]),
                requires_grad=False
            )

            self.rel_embed_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.out_dim]),
                requires_grad=False
            )

            self.ent_embed = nn.Embedding(
                n_entities, in_dim, max_norm=1, norm_type=2).to(device)
            self.rel_embed = nn.Embedding(
                n_relations, in_dim, max_norm=1, norm_type=2).to(device)

            nn.init.uniform_(self.ent_embed.weight.data, -
                             self.ent_embed_range.item(), self.ent_embed_range.item())
            nn.init.uniform_(self.rel_embed.weight.data, -
                             self.rel_embed_range.item(), self.rel_embed_range.item())

        self.input_drop = nn.Dropout(input_drop)

    def forward(self, triplets, ent_embed=None, rel_embed=None):
        N = self.n_entities

        if self.concat:
            h = torch.cat((
                self.ent_embed(triplets[:, 0]),
                self.rel_embed(triplets[:, 1]),
                self.ent_embed(triplets[:, 2])
            ), dim=1)
        else:
            h = torch.cat((
                ent_embed[triplets[:, 0]],
                rel_embed[triplets[:, 1]],
                ent_embed[triplets[:, 2]]
            ), dim=1)

        h = self.input_drop(h)
        c = self.a(h)
        b = -F.leaky_relu(self.a_2(c))
        e_b = torch.exp(b)

        temp = triplets.t()
        edges = torch.stack([temp[0], temp[2]])

        ebs = self.sparse_neighborhood_aggregation(
            edges, e_b, N, e_b.shape[0], 1)
        temp1 = e_b * c

        hs = self.sparse_neighborhood_aggregation(
            edges, temp1, N, e_b.shape[0], self.out_dim)

        ebs[ebs == 0] = 1e-12
        h_ent = hs / ebs

        index = triplets[:, 1]
        h_rel = scatter(temp1, index=index, dim=0, reduce="mean")

        if self.concat:
            return F.relu(h_ent), F.relu(h_rel)
        return h_ent, h_rel


class RotAttLayer(nn.Module):
    def __init__(self, n_ent, n_rel, in_dim, out_dim, n_heads=1, input_drop=0.5,
                 negative_rate=10, margin=6.0, epsilon=2.0, batch_size=None, device="cuda"):

        super().__init__()

        self.n_heads = n_heads
        self.device = device

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.margin = margin
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.negative_rate = negative_rate

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / in_dim]),
            requires_grad=False
        )

    def rotate(self, h, r, t, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
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
        score = score.norm(dim=0)

        score = self.margin - score.sum(dim=2)
        return score

    def forward(self, sample, ent_embed, rel_embed, mode="single"):
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                ent_embed, dim=0, index=sample[:, 0]).unsqueeze(1)
            relation = torch.index_select(
                rel_embed, dim=0, index=sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(
                ent_embed, dim=0, index=sample[:, 2]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(
                0), head_part.size(1)

            head = torch.index_select(
                ent_embed, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = torch.index_select(
                rel_embed, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(
                ent_embed, dim=0, index=tail_part[:, 2]).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(
                0), tail_part.size(1)

            head = torch.index_select(
                ent_embed, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(
                rel_embed, dim=0, index=head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(
                ent_embed, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        score = self.rotate(head, relation, tail, mode)

        return score
