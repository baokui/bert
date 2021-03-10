export CUDA_VISIBLE_DEVICES=5
nohup python -u bertSentvec.py >> log/bertSentvec.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python -u bertSentvec.py \
    --task_name=sort_new \
    --data_dir=/search/odin/guobk/vpa/vpa-studio-research/sort/data \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --train_batch_size=32 \
    --init_checkpoint=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt \
    --max_seq_length=128 \
    --num_train_epochs=3 \
    --do_predict=False \
    --do_train=True \
    --do_eval=False >> log/bertSentvec-new.log 2>&1 &


