export CUDA_VISIBLE_DEVICES=5
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
task_name=文字风格
mkdir -p model/label/$task_name
mkdir log
nohup python -u run_classifier.py \
    --data_dir=/search/odin/guobk/vpa/vpa-studio-research/labelClassify/DataLabel \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --output_dir=model/label/$task_name \
    --train_batch_size=32 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=512 \
    --do_train=True \
    --do_eval=True >> log/labelmodel-$task_name.log 2>&1 &