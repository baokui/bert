export CUDA_VISIBLE_DEVICES=5
nohup python -u bertSentvec.py >> log/bertSentvec.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python -u bertSentvec.py test >> log/bertSentvec-test.log 2>&1 &