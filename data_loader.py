import torch_geometric
from torch_geometric.datasets import TUDataset
from src.utils import mutag_preprocess

if __name__ == '__main__':
    graphs = TUDataset(root='./dataset/MUTAG' , name='MUTAG')
    mutag_preprocess()