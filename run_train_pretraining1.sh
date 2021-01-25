export CUDA_VISIBLE_DEVICES=6
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
nohup python -u run_classifier_multiClass.py \
    --input_file=data/vpa_tfrecord/vpaTst.tfrecord \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --train_batch_size=16 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --num_train_epochs=10 \
    --do_predict=False \
    --do_train=True \
    --do_eval=False \
    --output_dir=model/S2V >> log/train-S2V.log 2>&1 &