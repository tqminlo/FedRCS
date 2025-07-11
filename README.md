# FedMCS: Improving Federated Learning on Non-IID Data through Multiphase Client Selection
###### *This code is based on repository [FedNH](https://github.com/Yutong-Dai/FedNH.git)*
## Run experiments
    python main.py --purpose <name_wandb_project> --device <cuda_device> --yamlfile <path_yaml_file> --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta <dirichlet_distribution> --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --strategy <aggregation_method> --cs_method <client_selection_method> --global_seed <random_seed> --use_wandb True

Aggregation methods: **FedAvg**, **FedNH**, **FedROD**

Client Selection methods: **Random**, **Cluster1**, **FedMCS**

Other CS methods: **FedMCSv1** (old version), __FedMCS*__ (ranking use entropy score),  **FedMCSda**, **FedMCSag**, **FedMCSag** (other versions in ranking phase of *FedMCS*)

Example for *Cifar10-Dirichlet(0.3)-FedAvg-FedMCS* experiment:

    python main.py --purpose Cifar10Avg0.3 --device cuda:0 --yamlfile ./experiments/Cifar10_Conv2Cifar.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvg --cs_method FedMCS --global_seed 0 --use_wandb True