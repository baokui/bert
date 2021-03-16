export CUDA_VISIBLE_DEVICES=5
nohup python -u bertSentvec.py >> log/bertSentvec.log 2>&1 &

output_dir=/search/odin/guobk/data/bert_semantic/model/
mkdir $output_dir
data_dir=/search/odin/guobk/data/bert_semantic/data/
export CUDA_VISIBLE_DEVICES=4
nohup python -u bertSentvec.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --num_train_epochs=3 >> log/bertSentvec-new.log 2>&1 &


