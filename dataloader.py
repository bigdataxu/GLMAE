
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
def dataloader(args,device,root):
    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ])

    if args.dataset in {'arxiv', 'products', 'mag','proteins'}:
        from ogb.nodeproppred import PygNodePropPredDataset
        print('loading ogb dataset...')
        dataset = PygNodePropPredDataset(root=root, name=f'ogbn-{args.dataset}')
        if args.dataset in ['mag']:
            rel_data = dataset[0]
            # We are only interested in paper <-> paper relations.
            data = Data(
                x=rel_data.x_dict['paper'],
                edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                y=rel_data.y_dict['paper'])
            data = transform(data)
            split_idx = dataset.get_idx_split()
            data.train_nodes = split_idx['train']['paper']
            data.val_nodes = split_idx['valid']['paper']
            data.test_nodes = split_idx['test']['paper']
        else:
            data = transform(dataset[0])
            split_idx = dataset.get_idx_split()
            data.train_nodes = split_idx['train']
            data.val_nodes = split_idx['valid']
            data.test_nodes = split_idx['test']

    elif args.dataset in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, args.dataset)
        data = transform(dataset[0])

    elif args.dataset == 'Reddit':
        dataset = Reddit(osp.join(root, args.dataset))
        data = transform(dataset[0])
    elif args.dataset in {'Photo', 'Computers'}:
        dataset = Amazon(root, args.dataset)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif args.dataset in {'CS', 'Physics'}:
        dataset = Coauthor(root, args.dataset)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    else:
        raise ValueError(args.dataset)

    return data