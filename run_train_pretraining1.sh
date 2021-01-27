export CUDA_VISIBLE_DEVICES=5
BERT_BASE_DIR=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12
nohup python -u run_pretraining1.py \
    --input_file=data/vpa_tfrecord/vpaTrn.tfrecord \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --train_batch_size=16 \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --num_train_epochs=10 \
    --do_predict=False \
    --do_train=True \
    --do_eval=False \
    --max_predictions_per_seq=0 \
    --output_dir=model/S2V >> log/train-S2V.log 2>&1 &