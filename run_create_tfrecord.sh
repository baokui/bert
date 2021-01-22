nohup python -u create_pretraining_data1.py \
    --input_file=data/vpa.txt \
    --output_file=data/vpa_tfrecord/vpa.tfrecord \
    --vocab_file=/search/odin/guobk/vpa/roberta_zh/model/roberta_zh_l12/vocab.txt \
    --max_predictions_per_seq=0 >> log/create_tfdata_s2v.log 2>&1 &