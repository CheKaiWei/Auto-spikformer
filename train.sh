CUDA_VISIBLE_DEVICES=1 python train.py \
--mode retrain \
--model_cfg experiments/subnet/Spikformer.yaml \
--epochs 500 \
--experiment_description "retrain_spikformer baseline" 