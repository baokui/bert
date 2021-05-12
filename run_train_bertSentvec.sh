output_dir=/search/odin/guobk/data/bert_semantic/model/
mkdir $output_dir
data_dir=/search/odin/guobk/data/bert_semantic/data/
export CUDA_VISIBLE_DEVICES=0
nohup python -u bertSentvec.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --num_train_epochs=3 >> log/bertSentvec-new.log 2>&1 &

# bangxie-pretrain
# data pro
for((idx=0;idx<21;idx++))
do
output_dir=/search/odin/guobk/data/bert_semantic/pretrainData_tfrecord/train-$idx.tf_record
data_dir=/search/odin/guobk/data/bert_semantic/pretrainData/train-$idx.txt
export CUDA_VISIBLE_DEVICES=0
nohup python -u bertSentvec_datapro.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --num_train_epochs=3 >> log/bertSentvec-datapro-$idx.log 2>&1 &
done
# pretrain 
output_dir=/search/odin/guobk/data/bert_semantic/model3/
mkdir $output_dir
data_dir=/search/odin/guobk/data/bert_semantic/pretrainData_tfrecord/
export CUDA_VISIBLE_DEVICES=0
nohup python -u bertSentvec.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --number_examples=100000000 \
    --num_train_epochs=3 >> log/bertSentvec-new.log 2>&1 &
# bangxie-finetune
# data-prepro
mkdir /search/odin/guobk/data/bert_semantic/finetuneData_tfrecord
for((idx=0;idx<3;idx++))
do
output_dir=/search/odin/guobk/data/bert_semantic/finetuneData_tfrecord/train-$idx.tf_record
data_dir=/search/odin/guobk/data/bert_semantic/finetuneData/train-$idx.txt
export CUDA_VISIBLE_DEVICES=0
nohup python -u bertSentvec_datapro.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --num_train_epochs=3 >> log/bertSentvec-datapro-finetune-$idx.log 2>&1 &
done
# finetune
output_dir=/search/odin/guobk/data/bert_semantic/model3_finetune/
mkdir $output_dir
data_dir=/search/odin/guobk/data/bert_semantic/finetuneData_tfrecord/
export CUDA_VISIBLE_DEVICES=7
nohup python -u bertSentvec.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/data/bert_semantic/model3_backup/model.ckpt-601000 \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --number_examples=12000000 \
    --num_train_epochs=3 >> log/bertSentvec-new-finetune.log 2>&1 &

######################################################
# 单塔
# data-prepro
mkdir /search/odin/guobk/data/bert_semantic/singleBert_tfrecord
for((idx=0;idx<3;idx++))
do
output_dir=/search/odin/guobk/data/bert_semantic/singleBert_tfrecord/train-$idx.tf_record
data_dir=/search/odin/guobk/data/bert_semantic/finetuneData/train-$idx.txt
export CUDA_VISIBLE_DEVICES=0
nohup python -u bertSentvec_datapro_single.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --num_train_epochs=3 >> log/bertSentvec-datapro-single-$idx.log 2>&1 &
done
# finetune
output_dir=/search/odin/guobk/data/bert_semantic/model4/
mkdir $output_dir
data_dir=/search/odin/guobk/data/bert_semantic/singleBert_tfrecord/
export CUDA_VISIBLE_DEVICES=5
nohup python -u bertSentvec_single.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --number_examples=12000000 \
    --num_train_epochs=3 >> log/bertSentvec-new-single.log 2>&1 &