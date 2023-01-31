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
--experiment_description 'retrain from imagenet T' \
--opt sgd \
--weight-decay 1e-8 \
--lr 1e-2 \
--epochs 1000
# --eval \