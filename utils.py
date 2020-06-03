import torch 

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples



def generate_neighborhood(facts):
    unique = torch.cat([facts[:, 0], facts[:, 2]]).unique()
    neighborhood = {i: [] for i in range(len(unique))}
    for i in unique:
        neighborhood[i.item()] = torch.cat([
            facts[facts[:, 0] == i.item()], facts[facts[:, 2] == i.item()] 
        ])
    
    return neighborhood


def get_batch_neighbors(batch, n_map):
    unique = torch.cat([batch[:, 0], batch[:, 2]]).unique()
    return torch.cat(
        [n_map[i.item()] for i in unique]
    )