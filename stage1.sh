CUDA_VISIBLE_DEVICES=1 python train.py \
--mode super \
--model_cfg experiments/supernet/supernet-Spikformer_small_model_4_3t_new.yaml \
--epochs 1000 \
--experiment_description "stage1: auto-spikformer CIFAR100 search 3t new"