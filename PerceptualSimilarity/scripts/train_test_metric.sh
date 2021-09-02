
TRIAL=${1}
NET=${2}
GPU=${3}
mkdir checkpoints
mkdir checkpoints/${NET}_${TRIAL}
python ./train.py --train_trunk --datasets train/mix --use_gpu --net ${NET} --name ${NET}_${TRIAL} --gpu_ids ${GPU}
python ./test_dataset_model.py --use_gpu --net ${NET} --model_path ./checkpoints/${NET}_${TRIAL}/latest_net_.pth
