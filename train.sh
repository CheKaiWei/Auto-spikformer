CUDA_VISIBLE_DEVICES=0 python train.py \
--mode super \
--model_cfg experiments/supernet/supernet-Spikformer.yaml \
--resume logs/stage1_train_supernet/log_20230217_133001/checkpoint-1001.pth.tar