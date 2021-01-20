export CUDA_VISIBLE_DEVICES="2,3,4,5"
port=2022
ps -ef|grep $port|grep run_classifier_server.py|grep -v grep|awk  '{print "kill -9 " $2}' |sh
sleep 1s
nohup python -u run_classifier_server.py $port >> log/bert_cls-$port.log 2>&1 &