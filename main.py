from config import parser
import torch
import torch_geometric.transforms as T
import os.path as osp
from utils import set_seed, tab_printer
from dataloader import dataloader
from model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder,LPDecoder_ogb
from mask import MaskEdge, MaskPath,MaskumEdge,MaskdmEdge
from train_nodeclassification import train,train_nodeclas


def main():
    """set print"""
    try:
        args = parser.parse_args()
        print(tab_printer(args))
        if args.writer:
            fh = open('./log/{}_{}.txt'.format(args.dataset,args.layer), 'a')
            fh.write(tab_printer(args))
            fh.write('\r\n')
            fh.flush()
            fh.close()
    except:
        parser.print_help()
        exit(0)

    set_seed(args.seed)
    if args.device < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    root = osp.join('./data')

    data = dataloader(args, device, root)
    print(data)
    train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                        is_undirected=True,
                                                        split_labels=True,
                                                        add_negative_train_samples=True)(data)
    splits = dict(train=train_data, valid=val_data, test=test_data)

    if args.mask == 'Path':
        mask = MaskPath(p=args.p, num_nodes=data.num_nodes,
                        start=args.start,
                        walks_per_node=args.n,
                        walk_length=args.l)  # 0.7,2708,'node',3
    elif args.mask == 'Edge':
        mask = MaskEdge(p=args.p)
    else:
        mask = None

    encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                         num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                         bn=args.bn, layer=args.layer, activation=args.encoder_activation)
    edge_decoder = LPDecoder_ogb(args.hidden_channels,encoder_layer = args.encoder_layers,
                                 num_layers = args.edge_decoder,dropout=args.decoder_dropout,layer=args.layer)
    degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,encoder_layer = args.encoder_layers,
                                   num_layers=args.degree_decoder, dropout=args.decoder_dropout,layer=args.layer)

    model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)

    print(model)
    if args.writer:
        fh = open('./{}_{}.txt'.format(args.dataset, args.layer), 'a')
        fh.write(str(model))
        fh.write('\r\n')
        fh.flush()
        fh.close()

    train(model, data, splits, args, device=device)


if __name__ == "__main__":
    main()
