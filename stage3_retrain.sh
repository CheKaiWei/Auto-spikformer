python supernet_train.py \
--data-set CIFAR10 \
--data-path /home/hanjing/CHE/SpikeDHS_CLA/data \
--gp \
--change_qk \
--relative_position \
--mode retrain \
--dist-eval \
--cfg ./experiments/subnet/Spikformer.yaml \
--experiment_description 'stage3: spikformer version' \
--opt adamw \
--weight-decay 6e-2 \
--lr 5e-4 \
--min-lr 1e-5 \
--sched cosine \
--epochs 1000 \
--patch_size 4 \
--input-size 32 \
--batch-size 128 \
--warmup-epochs 20 \
--warmup-lr 1e-5 \
--epochs 300 \
--mixup 0.5 \
--cutmix 0 \
--remode const \


# --resume logs/pretrained_model/supernet-tiny.pth \
# --eval \