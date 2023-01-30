CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python supernet_train.py \
--data-set CIFAR10 \
--data-path /home/hanjing/CHE/SpikeDHS_CLA/data \
--gp \
--change_qk \
--relative_position \
--mode super \
--dist-eval \
--cfg ./experiments/supernet/supernet-T.yaml \
--epochs 50 \
--warmup-epochs 10 \
--output logs \
--batch-size 32