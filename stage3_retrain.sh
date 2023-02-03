CUDA_VISIBLE_DEVICES=1 python supernet_train.py \
--data-set CIFAR10 \
--data-path /home/hanjing/CHE/SpikeDHS_CLA/data \
--gp \
--change_qk \
--relative_position \
--mode retrain \
--dist-eval \
--cfg ./experiments/subnet/AutoFormer-T.yaml \
--experiment_description 'stage3: auto-spikformer structure=T' \
--opt adamw \
--weight-decay 6e-2 \
--lr 5e-4 \
--min-lr 1e-5 \
--sched cosine \
--patch_size 4 \
--input-size 32 \
--batch-size 128 \
--warmup-epochs 20 \
--warmup-lr 1e-5 \
--epochs 300 \
--mixup 0.5 \
--mixup-off-epoch 200 \
--cutmix 0 \
--remode const \
--color-jitter 0 \


# --resume logs/pretrained_model/supernet-tiny.pth \
# --eval \