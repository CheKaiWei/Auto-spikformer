CUDA_VISIBLE_DEVICES=0 python train.py \
--mode super \
--model_cfg experiments/supernet/supernet-Spikformer.yaml \
--resume /home/hanjing/CHE/Spikformer/logs/stage1_train_supernet/log_20230213_175210/checkpoint-305.pth.tar