from torch_geometric.datasets import TUDataset
import argparse
from src.utils import dataset_preprocess, dataset_preprocess_without_edge_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocess the data set")
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG), and also can be "PROTEINS"')
    args = parser.parse_args()
    graphs = TUDataset(root=f"dataset/{args.dataset}" , name=args.dataset)
    
    if args.dataset == "MUTAG":
        dataset_preprocess(args.dataset)

    if args.dataset == "PROTEINS":
        dataset_preprocess_without_edge_label(args.dataset)
