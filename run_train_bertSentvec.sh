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
# pretrain: 修改激活函数，冻结部分参数
output_dir=/search/odin/guobk/data/bert_semantic/model3_new/
mkdir $output_dir
data_dir=/search/odin/guobk/data/bert_semantic/pretrainData_tfrecord/
export CUDA_VISIBLE_DEVICES=4
nohup python -u bertSentvec.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --number_examples=100000000 \
    --num_train_epochs=3 >> log/bertSentvec-new-new.log 2>&1 &

#####################
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
# finetune: 修改激活函数，冻结部分参数
output_dir=/search/odin/guobk/data/bert_semantic/model3_finetune_new/
mkdir $output_dir
data_dir=/search/odin/guobk/data/bert_semantic/finetuneData_tfrecord/
export CUDA_VISIBLE_DEVICES=3
nohup python -u bertSentvec.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/data/bert_semantic/model3_backup/model.ckpt-601000 \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --number_examples=12000000 \
    --num_train_epochs=3 >> log/bertSentvec-new-finetune-new.log 2>&1 &
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
# finetune-origin
output_dir=/search/odin/guobk/data/bert_semantic/model5/
mkdir $output_dir
data_dir=/search/odin/guobk/data/bert_semantic/finetuneData/
train_file=/search/odin/guobk/data/bert_semantic/singleBert_tfrecord/train-0.tf_record
export CUDA_VISIBLE_DEVICES=5
nohup python -u run_classifier0.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --train_file=$train_file \
    --bert_config_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/data/bert_semantic/model5/model.ckpt-228000 \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --do_train=True \
    --task_name=cola \
    --nb_train_examples=4000000 \
    --num_train_epochs=3 >> log/bertSentvec-new-single-model5.log 2>&1 &




#####################
# 新数据0422-0518
# data-prepro
mkdir /search/odin/guobk/data/bert_semantic/finetuneData_tfrecord
mkdir log
for((idx=0;idx<8;idx++))
do
output_dir=/search/odin/guobk/data/bert_semantic/finetuneData_tfrecord/train-$idx.tf_record
data_dir=/search/odin/guobk/data/bert_semantic/finetuneData_new/train-$idx.txt
export CUDA_VISIBLE_DEVICES=0
nohup python -u bertSentvec_datapro.py \
    --output_dir=$output_dir \
    --data_dir=$data_dir \
    --bert_config_file=/search/odin/guobk/data/model/roberta_zh_l12/bert_config.json \
    --vocab_file=/search/odin/guobk/data/model/roberta_zh_l12/vocab.txt \
    --init_checkpoint=/search/odin/guobk/data/model/roberta_zh_l12/bert_model.ckpt \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --num_train_epochs=3 >> log/bertSentvec-datapro-finetune-$idx.log 2>&1 &
done
