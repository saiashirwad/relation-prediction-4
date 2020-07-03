from utils import * 
from dataloader import * 
import os 

from torch.utils.data import DataLoader

dataset = "FB15k-237"
device = "cuda"
batch_size = 100
negative_sample_size = 10

data_path = f"data/{dataset}"
with open(os.path.join(data_path, 'entities.dict')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(os.path.join(data_path, 'relations.dict')) as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)

n_ent = len(entity2id)
n_rel = len(relation2id)

train_triplets = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
valid_triplets = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
test_triplets = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
all_true_triplets = train_triplets + valid_triplets + test_triplets

facts = torch.Tensor(train_triplets).to(torch.long).to(device)

train_dataloader_head = DataLoader(
    TrainDataset(train_triplets, n_ent, n_rel, negative_sample_size, 'head-batch'), 
    batch_size=batch_size,
    shuffle=True, 
    collate_fn=TrainDataset.collate_fn
)
train_dataloader_tail = DataLoader(
    TrainDataset(train_triplets, n_ent, n_rel, negative_sample_size, 'tail-batch'), 
    batch_size=batch_size,
    shuffle=True, 
    collate_fn=TrainDataset.collate_fn
)
train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

facts = torch.Tensor(train_triplets).to(torch.long).to(device)
edges = torch.stack([facts[:, 0], facts[:, 2]])
edges =  [[int(e.item()) for e in edges[i]] for i in range(2)]      

keys = list(set(edges[0]))
adj = {key: [] for key in keys}

for i in range(len(edges[0])):
    adj[edges[0][i]].append(edges[1][i])

def generate_neighborhood(facts):
    unique = torch.cat([facts[:, 0], facts[:, 2]]).unique()
    neighborhood = {i: [] for i in range(len(unique))}
    for i in unique:
        neighborhood[i.item()] = torch.cat([
            facts[facts[:, 0] == i.item()], facts[facts[:, 2] == i.item()] 
        ])
    
    return neighborhood



