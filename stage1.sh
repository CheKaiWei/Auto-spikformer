CUDA_VISIBLE_DEVICES=0 python train.py \
--mode super \
--model_cfg experiments/supernet/supernet-Spikformer_small_model_1.yaml \
--epochs 1000 \
--experiment_description "stage1: auto-spikformer CIFAR100 small"