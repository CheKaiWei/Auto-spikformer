python evolution.py \
--data-path /home/hanjing/CHE/SpikeDHS_CLA/data \
--gp \
--change_qk \
--relative_position \
--dist-eval \
--cfg ./experiments/supernet/supernet-T.yaml --resume logs/checkpoint.pth \
--data-set CIFAR10 \
--min-param-limits 1 --param-limits 100 \
--output_dir logs/search/
