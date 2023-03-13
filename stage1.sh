CUDA_VISIBLE_DEVICES=1 python train.py \
--mode super \
--model_cfg experiments/supernet/supernet-Spikformer_small_model_5_3tsmall_new.yaml \
--epochs 2000 \
--experiment_description "stage1: auto-spikformer CIFAR100 search 3t and small new"