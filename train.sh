CUDA_VISIBLE_DEVICES=1 python train.py \
--mode retrain \
--model_cfg experiments/subnet/Spikformer.yaml \
--epochs 500 \
--experiment_description "retrain Spikformer_small_model_1 small learning rate" \
--warmup-epochs 20 \
--warmup-lr 1e-7 \
--lr 1e-5 \
--min-lr 1e-6 \
# --resume /home/hanjing/CHE/Spikformer/logs/stage1_train_supernet/log_20230222_100148/model_best.pth.tar \