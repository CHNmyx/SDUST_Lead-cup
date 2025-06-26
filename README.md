# SAGE

## Setup

### Environments
Implementing environment:  
- numpy = 1.23.1  
- pytorch = 1.10.1  
- torch_geometric = 2.0.4  
- torch_scatter = 2.0.9  
- torch_sparse = 0.6.13  

### Training
parser.add_argument("--dataset", type=str, default="DGraphFin")
parser.add_argument("--model", type=str, default="GEARSage")
parser.add_argument("--device", type=int, default=0)  # 默认使用 GPU 0
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--hiddens", type=int, default=96)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=5000)  # 默认批次大小为 5000

### 直接使用
python3 main.py
python3 main.py --batch_size 2000 --device 0

# Graph Network Simulator (GNS)
In this project, we use machine learning framework Graph Neural Network-based Simulators (GNS) as a surrogate model to predict granular and fluid flow dynamics.

## Run GNS

> Training GNS on simulation data
```shell
python3 -m train --data_path="datasets/" --model_path="models/" --output_path="outputs/" -ntraining_steps=100
```

> Resume training

To resume training specify `model_file` and `train_state_file`:

```shell
python3 -m train --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>"  --model_file="model.pt" --train_state_file="train_state.pt" -ntraining_steps=100
```

> Rollout prediction
```shell
python3 -m train --mode="rollout" --data_path="<input-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" --model_file="model.pt" --train_state_file="train_state.pt"
```

> Render
```shell
python3 -m render_rollout --output_mode="gif" --rollout_dir="<path-containing-rollout-file>" --rollout_name="<name-of-rollout-file>"
```


