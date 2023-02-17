python search.py \
--model_cfg experiments/supernet/supernet-Spikformer.yaml \
 --resume logs/stage1_train_supernet/log_20230215_163857/checkpoint-501.pth.tar \
--min-param-limits 1 --param-limits 100 