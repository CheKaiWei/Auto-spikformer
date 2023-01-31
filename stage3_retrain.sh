python supernet_train.py \
--data-set CIFAR10 \
--data-path /home/hanjing/CHE/SpikeDHS_CLA/data \
--gp \
--change_qk \
--relative_position \
--mode retrain \
--dist-eval \
--cfg ./experiments/subnet/AutoFormer-T.yaml \
--resume logs/pretrained_model/supernet-tiny.pth \
--eval \
--output_dir logs/stage3_retrain/