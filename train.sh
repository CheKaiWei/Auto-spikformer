CUDA_VISIBLE_DEVICES=1 python train.py \
--mode retrain \
--model_cfg experiments/subnet/Spikformer.yaml \
--experiment_description "auto-spikformer CIFAR100 baseline" \
# --resume /home/hanjing/CHE/Spikformer/logs/stage1_train_supernet/log_20230222_100148/model_best.pth.tar \