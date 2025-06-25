import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from feat_func import data_process
from models import GEARSage
from utils import DGraphFin
from utils.evaluator import Evaluator
from utils.utils import prepare_folder


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, data, optimizer):
    model.train()

    optimizer.zero_grad()
    neg_idx = data.train_neg[
        torch.randperm(data.train_neg.size(0))[: data.train_pos.size(0)]
    ]
    train_idx = torch.cat([data.train_pos, neg_idx], dim=0)

    nodeandneighbor, edge_index, node_map, mask = k_hop_subgraph(
        train_idx, 3, data.edge_index, relabel_nodes=True, num_nodes=data.x.size(0)
    )

    out = model(
        data.x[nodeandneighbor],
        edge_index,
        data.edge_attr[mask],
        data.edge_timestamp[mask],
        data.edge_direct[mask],
    )
    loss = F.nll_loss(out[node_map], data.y[train_idx])
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    optimizer.step()
    torch.cuda.empty_cache()
    return loss.item()


@torch.no_grad()
def test(model, data, batch_size=10000):
    model.eval()
    
    # 获取节点总数
    num_nodes = data.x.size(0)
    # 初始化预测结果张量
    y_pred = torch.zeros(num_nodes, model.convs[-1].out_channels).to(data.x.device)
    
    # 分批处理节点
    for i in range(0, num_nodes, batch_size):
        # 获取当前批次的节点索引
        batch_nodes = torch.arange(i, min(i + batch_size, num_nodes)).to(data.x.device)
        
        # 提取子图
        nodeandneighbor, edge_index, node_map, mask = k_hop_subgraph(
            batch_nodes, 2, data.edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        
        # 前向传播
        batch_out = model(
            data.x[nodeandneighbor],
            edge_index,
            data.edge_attr[mask],
            data.edge_timestamp[mask],
            data.edge_direct[mask],
        )
        
        # 将结果映射回原始节点索引
        y_pred[batch_nodes] = batch_out[node_map].exp()
        
        # 清理缓存减少内存占用
        torch.cuda.empty_cache()
    
    return y_pred


def main():
    parser = argparse.ArgumentParser(description="GEARSage for DGraphFin Dataset")
    parser.add_argument("--dataset", type=str, default="DGraphFin")
    parser.add_argument("--model", type=str, default="GEARSage")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hiddens", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=5000, help="批处理大小，用于控制内存使用")
    parser.add_argument("--cpu", action="store_true", help="使用CPU进行推理，避免GPU内存不足")

    args = parser.parse_args()
    print(args)

    # 如果指定使用CPU或CUDA不可用，则使用CPU
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
        print("使用CPU进行计算")
    else:
        device = f"cuda:{args.device}"
        print(f"使用GPU {args.device} 进行计算")
    device = torch.device(device)
    model_dir = prepare_folder(args.dataset, args.model)
    print("model_dir:", model_dir)
    set_seed(42)
    dataset = DGraphFin(root="./dataset", name="DGraphFin")

    nlabels = 2

    data = dataset[0]

    split_idx = {
        "train": data.train_mask,
        "valid": data.valid_mask,
        "test": data.test_mask,
    }

    data = data_process(data).to(device)
    train_idx = split_idx["train"].to(device)

    data.train_pos = train_idx[data.y[train_idx] == 1]
    data.train_neg = train_idx[data.y[train_idx] == 0]
    model = GEARSage(
        in_channels=data.x.size(-1),
        hidden_channels=args.hiddens,
        out_channels=nlabels,
        num_layers=args.layers,
        dropout=args.dropout,
        activation="elu",
        bn=True,
    ).to(device)

    print(f"Model {args.model} initialized")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    best_auc = 0.0
    evaluator = Evaluator("auc")
    y_train, y_valid = data.y[data.train_mask], data.y[data.valid_mask]
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)
        out = test(model, data, batch_size=args.batch_size)
        preds_train, preds_valid = out[data.train_mask], out[data.valid_mask]
        train_auc = evaluator.eval(y_train, preds_train)["auc"]
        valid_auc = evaluator.eval(y_valid, preds_valid)["auc"]

        if valid_auc >= best_auc:
            best_auc = valid_auc
            torch.save(model.state_dict(), model_dir + "model.pt")
            preds = out[data.test_mask].cpu().numpy()
        print(
            f"Epoch: {epoch:02d}, "
            f"Loss: {loss:.4f}, "
            f"Train: {train_auc:.2%}, "
            f"Valid: {valid_auc:.2%},"
            f"Best: {best_auc:.4%},"
        )

    test_auc = evaluator.eval(data.y[data.test_mask], preds)["auc"]
    print(f"test_auc: {test_auc}")


if __name__ == "__main__":
    main()
