# 智能计算创新设计赛(先导杯)山东科技大学校内赛-参赛作品
> Project Members: MenYX, FengX, ShengHF, ZhaoLY
# 面向动态系统的图神经网络建模方法研究及边属性时序融合与颗粒流动力学模拟
# Research on Graph Neural Network Modeling Method for Dynamic Systems and Temporal fusion of edge properties and simulation of particle flow dynamics

# Environments
Implementing environment:  
- numpy = 1.23.1  
- pytorch = 1.10.1  
- torch_geometric = 2.0.4  
- torch_scatter = 2.0.9  
- torch_sparse = 0.6.13  


# SAGE
# 项目出发点
本项目研究基于GraphSAGE模型，针对动态图数据处理中的挑战，提出了改进的节点特征表达方法。通过融入时序信息，增强模型对网络动态演化特征的捕获能力。同时，本研究还采用多种优化策略，以提升模型的整体性能。这些改进不仅有助于更好地理解和预测动态网络的演化规律，也为实际应用提供了更有效的解决方案。
## Setup
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
# 项目出发点
本项目希望利用基于图神经网络的模拟器（Graph Neural Network-based Simulators，GNS）这一机器学习框架，作为替代模型来预测颗粒流与流体流动的动态行为，本项目在Sanchez-Gonzalez（2020）与Choi（2023）提出的GNS框架基础上加以扩展，完成了以下任务：
1. 通过模拟颗粒柱或水滴塌落过程以及其与障碍物的相互作用，展示GNN在建模颗粒流动力学方面的有效性。
2. 探讨这些不同的GNN聚合操作之间对模型来预测颗粒流与流体流动的结果差异，并对这种差异寻找一种解释。
## Setup
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

