python ../main.py  --purpose BrainTumorResnetAvgb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvg --global_seed 0

python ../main.py  --purpose BrainTumorResnetAvgb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS10 --global_seed 0

#python ../main.py  --purpose BrainTumorResnetAvgb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvg --global_seed 1000

#python ../main.py  --purpose BrainTumorResnetAvgb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS10 --global_seed 1000

#python ../main.py  --purpose BrainTumorResnetAvgb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvg --global_seed 2000

#python ../main.py  --purpose BrainTumorResnetAvgb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS10 --global_seed 2000

python ../main.py  --purpose BrainTumorResnetAvgb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvg --global_seed 1000

python ../main.py  --purpose BrainTumorResnetAvgb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS10 --global_seed 1000

#python ../main.py  --purpose BrainTumorResnetAvgb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvg --global_seed 1000

#python ../main.py  --purpose BrainTumorResnetAvgb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS10 --global_seed 1000

#python ../main.py  --purpose BrainTumorResnetAvgb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvg --global_seed 2000

#python ../main.py  --purpose BrainTumorResnetAvgb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS10 --global_seed 2000




#python ../main.py  --purpose BrainTumorResnetb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNH --global_seed 0

#python ../main.py  --purpose BrainTumorResnetb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNHCS10 --global_seed 0

#python ../main.py  --purpose BrainTumorResnetb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNHCS7 --global_seed 0

#python ../main.py  --purpose BrainTumorResnetb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNH --global_seed 0

#python ../main.py  --purpose BrainTumorResnetb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNHCS10 --global_seed 0

#python ../main.py  --purpose BrainTumorResnetb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNHCS7 --global_seed 0


#python ../main.py  --purpose BrainTumorResnetb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNH --global_seed 1000

#python ../main.py  --purpose BrainTumorResnetb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNHCS10 --global_seed 1000

#python ../main.py  --purpose BrainTumorResnetb1.0 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNHCS7 --global_seed 1000

#python ../main.py  --purpose BrainTumorResnetb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNH --global_seed 1000

#python ../main.py  --purpose BrainTumorResnetb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNHCS10 --global_seed 1000

#python ../main.py  --purpose BrainTumorResnetb0.3 --device cuda:1 --yamlfile ./BrainMRI.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedNHCS7 --global_seed 1000


#python ../main.py  --purpose ICHResnetb1.0 --device cuda:1 --yamlfile ./ICH.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --strategy FedNH --use_wandb True --global_seed 0

#python ../main.py  --purpose ICHResnetb1.0 --device cuda:1 --yamlfile ./ICH.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --strategy FedNHCS10 --use_wandb True --global_seed 0

#python ../main.py  --purpose ICHResnetb1.0 --device cuda:1 --yamlfile ./ICH.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --strategy FedNHCS7 --use_wandb True --global_seed 0

#python ../main.py  --purpose ICHResnetb1.0 --device cuda:1 --yamlfile ./ICH.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --strategy FedNH --use_wandb True --global_seed 1000

#python ../main.py  --purpose ICHResnetb1.0 --device cuda:1 --yamlfile ./ICH.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --strategy FedNHCS10 --use_wandb True --global_seed 1000

#python ../main.py  --purpose ICHResnetb1.0 --device cuda:1 --yamlfile ./ICH.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --strategy FedNHCS7 --use_wandb True --global_seed 1000


#python ../main.py  --purpose Cifar10Avgb1.0 --device cuda:1 --yamlfile ./Cifar10_Conv2Cifar.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvg --global_seed 0 &

#python ../main.py  --purpose Cifar10Avgb1.0 --device cuda:1 --yamlfile ./Cifar10_Conv2Cifar.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS7 --global_seed 0 &

#python ../main.py  --purpose Cifar10Avgb1.0 --device cuda:1 --yamlfile ./Cifar10_Conv2Cifar.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS10 --global_seed 0 &

#python ../main.py  --purpose Cifar10Avgb1.0 --device cuda:1 --yamlfile ./Cifar10_Conv2Cifar.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvg --global_seed 3000 &

#python ../main.py  --purpose Cifar10Avgb1.0 --device cuda:1 --yamlfile ./Cifar10_Conv2Cifar.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS7 --global_seed 3000 &

#python ../main.py  --purpose Cifar10Avgb1.0 --device cuda:1 --yamlfile ./Cifar10_Conv2Cifar.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS10 --global_seed 3000

#python ../main.py  --purpose Cifar10Avgb1.0 --device cuda:1 --yamlfile ./Cifar10_Conv2Cifar.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvg --global_seed 5000 &

#python ../main.py  --purpose Cifar10Avgb1.0 --device cuda:1 --yamlfile ./Cifar10_Conv2Cifar.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS7 --global_seed 5000 &

#python ../main.py  --purpose Cifar10Avgb1.0 --device cuda:1 --yamlfile ./Cifar10_Conv2Cifar.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --use_wandb True --strategy FedAvgCS10 --global_seed 5000


#python ../main.py  --purpose ICHb1.0 --device cuda:1 --yamlfile ./ICH.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --strategy FedNHCS7 --use_wandb True --global_seed 0

#python ../main.py  --purpose ICHb0.3 --device cuda:1 --yamlfile ./ICH.yaml --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 --strategy FedNHCS7 --use_wandb True --global_seed 0
