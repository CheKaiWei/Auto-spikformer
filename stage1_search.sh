CUDA_VISIBLE_DEVICES=0 python train.py \
--mode super \
--model_cfg experiments/supernet/supernet-Spikformer_small.yaml \
--resume 'logs/stage1_train_supernet/log_20230221_230318/model_best.pth.tar' \
--epochs 1000