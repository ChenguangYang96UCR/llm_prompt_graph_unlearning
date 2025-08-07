from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import WebKB
import argparse
from src.utils import dataset_preprocess, dataset_preprocess_without_edge_label, WebKB_preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocess the data set")
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG), and also can be "PROTEINS", "BZR", "COX2", "ENZYMES", "cornell", "wisconsin", "texas"')
    args = parser.parse_args()

    if args.dataset == "MUTAG" or args.dataset == "PROTEINS" or args.dataset == "BZR" or args.dataset == "COX2" or args.dataset == "ENZYMES":
        graphs = TUDataset(root=f"dataset/" , name=args.dataset)

    if args.dataset == "cornell" : 
        dataset = WebKB(root=f"dataset/", name='Cornell')

    if args.dataset == "wisconsin" : 
        dataset = WebKB(root=f"dataset/", name='Wisconsin')

    if args.dataset == "texas" : 
        dataset = WebKB(root=f"dataset/", name='Texas')
    
    if args.dataset == "MUTAG":
        dataset_preprocess(args.dataset)

    if args.dataset == "PROTEINS":
        dataset_preprocess_without_edge_label(args.dataset, True)

    if args.dataset == "BZR":
        dataset_preprocess_without_edge_label(args.dataset, False)

    if args.dataset == "COX2":
        dataset_preprocess_without_edge_label(args.dataset, False)

    if args.dataset == "ENZYMES":
        dataset_preprocess_without_edge_label(args.dataset, True)

    if args.dataset == "cornell" or args.dataset == 'wisconsin' or args.dataset == "texas" : 
        WebKB_preprocess(args.dataset)
