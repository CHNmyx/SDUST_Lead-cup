
### SAGE
## Environments
Implementing environment:  
- numpy = 1.23.1  
- pytorch = 1.10.1  
- torch_geometric = 2.0.4  
- torch_scatter = 2.0.9  
- torch_sparse = 0.6.13  


## Training

parser.add_argument("--dataset", type=str, default="DGraphFin")
parser.add_argument("--model", type=str, default="GEARSage")
parser.add_argument("--device", type=int, default=0)  # 默认使用 GPU 0
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--hiddens", type=int, default=96)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=5000)  # 默认批次大小为 5000


直接使用
python3 main.py

python3 main.py --batch_size 2000 --device 0
