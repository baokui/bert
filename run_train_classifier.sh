export CUDA_VISIBLE_DEVICES=4
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
task_name=使用场景P0
mkdir -p model/label/$task_name
mkdir log
nohup python -u run_classifier.py \
    --data_dir=/search/odin/guobk/vpa/vpa-studio-research/labelClassify/DataLabel \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --output_dir=model/label/$task_name \
    --train_batch_size=8 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --do_predict=True \
    --do_train=False \
    --do_eval=False >> log/labelmodel-test-$task_name.log 2>&1 &
export CUDA_VISIBLE_DEVICES=5
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
task_name=表达对象P0
mkdir -p model/label/$task_name
mkdir log
nohup python -u run_classifier.py \
    --data_dir=/search/odin/guobk/vpa/vpa-studio-research/labelClassify/DataLabel \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --output_dir=model/label/$task_name \
    --train_batch_size=8 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --do_predict=True \
    --do_train=False \
    --do_eval=False >> log/labelmodel-test-$task_name.log 2>&1 &
export CUDA_VISIBLE_DEVICES=6
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
task_name=表达者性别倾向P0
mkdir -p model/label/$task_name
mkdir log
nohup python -u run_classifier.py \
    --data_dir=/search/odin/guobk/vpa/vpa-studio-research/labelClassify/DataLabel \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --output_dir=model/label/$task_name \
    --train_batch_size=8 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --do_predict=True \
    --do_train=False \
    --do_eval=False >> log/labelmodel-test-$task_name.log 2>&1 &
#export CUDA_VISIBLE_DEVICES=4
#BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
#task_name=文字风格
#mkdir -p model/label/$task_name
#mkdir log
#nohup python -u run_classifier.py \
#    --data_dir=/search/odin/guobk/vpa/vpa-studio-research/labelClassify/DataLabel \
#    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#    --task_name=$task_name \
#    --vocab_file=$BERT_BASE_DIR/vocab.txt \
#    --output_dir=model/label/$task_name \
#    --train_batch_size=8 \
#    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#    --max_seq_length=512 \
#    --do_train=False \
#    --do_eval=False \
#    --do_predict=True >> log/labelmodel-predict-$task_name.log 2>&1 &
export CUDA_VISIBLE_DEVICES=6
BERT_BASE_DIR=/search/odin/guobk/data/model/roberta_zh_l12
nohup python -u run_classifier_multiClass.py \
    --data_dir=/search/odin/guobk/vpa/vpa-studio-research/labelClassify/DataLabel \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --train_batch_size=16 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --num_train_epochs=10 \
    --do_predict=False \
    --do_train=True \
    --do_eval=False >> log/labelmodel-multiClass-train.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
nohup python -u run_classifier_multiClass1.py \
    --data_dir=/search/odin/guobk/vpa/vpa-studio-research/labelClassify/DataLabel \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --train_batch_size=16 \
    --init_checkpoint=model/S2V/model.ckpt-21000 \
    --max_seq_length=128 \
    --num_train_epochs=10 \
    --do_predict=False \
    --do_train=True \
    --do_eval=False >> log/labelmodel-multiClass-train1.log 2>&1 &
###################################################################################
### 句库新标签
export CUDA_VISIBLE_DEVICES=7
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
task_name=newlabel
output_dir=/search/odin/guobk/data/labels/data_new/model/
mkdir -p $output_dir
mkdir log
nohup python -u run_classifier.py \
    --data_dir=/search/odin/guobk/data/labels/data_new \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=/search/odin/guobk/data/labels/data_new/vocab.txt \
    --output_dir=$output_dir \
    --train_batch_size=8 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --do_predict=False \
    --do_train=True \
    --num_train_epochs=10 \
    --do_eval=False >> log/labelmodel-train-$task_name.log 2>&1 &

export CUDA_VISIBLE_DEVICES=7
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
task_name=newlabel
output_dir=/search/odin/guobk/data/labels/data_new/model/
mkdir -p $output_dir
mkdir log
nohup python -u run_classifier.py \
    --data_dir=/search/odin/guobk/data/labels/data_new \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=/search/odin/guobk/data/labels/data_new/vocab.txt \
    --output_dir=$output_dir \
    --train_batch_size=8 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --do_predict=True \
    --do_train=False \
    --num_train_epochs=10 \
    --do_eval=False >> log/labelmodel-predict-$task_name.log 2>&1 &

#######################################################################
### 句库新标签
data=doubi
export CUDA_VISIBLE_DEVICES=7
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
task_name=newlabel
output_dir=/search/odin/guobk/data/labels/data_new/bimodel/model_$data/
mkdir -p $output_dir
mkdir log
nohup python -u run_bi_classifier.py \
    --data_dir=/search/odin/guobk/data/labels/data_new/subdata/data_$data \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=/search/odin/guobk/data/labels/data_new/vocab.txt \
    --output_dir=$output_dir \
    --train_batch_size=8 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --do_predict=False \
    --do_train=True \
    --num_train_epochs=10 \
    --do_eval=False >> log/labelmodel-train-$task_name-$data.log 2>&1 &
data=gufeng
export CUDA_VISIBLE_DEVICES=4
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
task_name=newlabel
output_dir=/search/odin/guobk/data/labels/data_new/bimodel/model_$data/
mkdir -p $output_dir
mkdir log
nohup python -u run_bi_classifier.py \
    --data_dir=/search/odin/guobk/data/labels/data_new/subdata/data_$data \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=/search/odin/guobk/data/labels/data_new/vocab.txt \
    --output_dir=$output_dir \
    --train_batch_size=8 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --do_predict=False \
    --do_train=True \
    --num_train_epochs=10 \
    --do_eval=False >> log/labelmodel-train-$task_name-$data.log 2>&1 &
###################################################################################
### 句库新标签
export CUDA_VISIBLE_DEVICES=5
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
task_name=newlabel
output_dir=/search/odin/guobk/data/labels/data_new/model_layer2/
mkdir -p $output_dir
mkdir log
nohup python -u run_classifier1.py \
    --data_dir=/search/odin/guobk/data/labels/data_new \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=/search/odin/guobk/data/labels/data_new/vocab.txt \
    --output_dir=$output_dir \
    --train_batch_size=8 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --do_predict=False \
    --do_train=True \
    --num_train_epochs=10 \
    --learning_rate=0.1 \
    --do_eval=False >> log/labelmodel-train-$task_name-layer2.log 2>&1 &

###################################################################################
### 句库新标签 - max_seq_length=32
export CUDA_VISIBLE_DEVICES=7
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
task_name=newlabel
output_dir=/search/odin/guobk/data/labels/data_new/model32/
mkdir -p $output_dir
mkdir log
nohup python -u run_classifier.py \
    --data_dir=/search/odin/guobk/data/labels/data_new \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=/search/odin/guobk/data/labels/data_new/vocab.txt \
    --output_dir=$output_dir \
    --train_batch_size=8 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=32 \
    --do_predict=False \
    --do_train=True \
    --num_train_epochs=10 \
    --do_eval=False >> log/labelmodel-train-$task_name-32.log 2>&1 &

### 句库新标签 - max_seq_length=32 - pretrained with juku content
export CUDA_VISIBLE_DEVICES=7
task_name=newlabel
output_dir=/search/odin/guobk/data/labels/data_new/model_pretrain/finetune/
mkdir -p $output_dir
nohup python -u run_classifier.py \
    --data_dir=/search/odin/guobk/data/labels/data_new \
    --bert_config_file=/search/odin/guobk/data/labels/data_new/model_pretrain/bert_config.json \
    --task_name=$task_name \
    --vocab_file=/search/odin/guobk/data/labels/data_new/model_pretrain/vocab.txt \
    --output_dir=$output_dir \
    --train_batch_size=8 \
    --init_checkpoint=/search/odin/guobk/data/labels/data_new/model_pretrain/ckpt/model.ckpt-84000 \
    --max_seq_length=32 \
    --do_predict=False \
    --do_train=True \
    --num_train_epochs=10 \
    --do_eval=False >> log/labelmodel-train-$task_name-32-pre.log 2>&1 &

#################################
# 质量模型
export CUDA_VISIBLE_DEVICES=7
task_name=textClassify
output_dir=/search/odin/guobk/data/AiWriter/model/ckpt/
mkdir -p $output_dir
BERT_BASE_DIR=/search/odin/guobk/data/model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
nohup python -u run_classifier.py \
    --data_dir=/search/odin/guobk/data/AiWriter/Content/DataQuality/data_new \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --output_dir=$output_dir \
    --train_batch_size=8 \
    --init_checkpoint=$BERT_BASE_DIR/ckpt/bert_model.ckpt \
    --save_checkpoints_steps=10000 \
    --max_seq_length=128 \
    --do_predict=True \
    --do_train=True \
    --do_eval=True \
    --nb_classes=4 \
    --num_train_epochs=5 >> log/DataQuality_model.log 2>&1 &