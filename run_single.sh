#PYTHON="/data/anaconda/envs/pytorch1.7.1/bin/python"
#CUDA_VISIBLE_DEVICES=0
#CUDA_LAUNCH_BLOCKING=1
save_name="test"
#nohup python tools/train.py > log/$save_name.log 2>&1&
nohup python tools/train_cls.py > log/$save_name.log 2>&1&
tail -f log/$save_name.log