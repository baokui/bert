gpu=2
for((i=0;i<4;i++))
do
path_source="/search/odin/guobk/data/dabaigou/000$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/000$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-0$i 2>&1 
done
i=4
path_source="/search/odin/guobk/data/dabaigou/000$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/000$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-0$i 2>&1 &
########################
gpu=3
for((i=5;i<9;i++))
do
path_source="/search/odin/guobk/data/dabaigou/000$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/000$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-0$i 2>&1 
done
i=9
path_source="/search/odin/guobk/data/dabaigou/000$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/000$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-0$i 2>&1 &
########################
gpu=4
for((i=10;i<14;i++))
do
path_source="/search/odin/guobk/data/dabaigou/00$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$i 2>&1 
done
i=14
path_source="/search/odin/guobk/data/dabaigou/00$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$i 2>&1 &
########################
gpu=5
for((i=15;i<19;i++))
do
path_source="/search/odin/guobk/data/dabaigou/00$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$i 2>&1 
done
i=19
path_source="/search/odin/guobk/data/dabaigou/00$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$i 2>&1 &
########################
gpu=6
for((i=20;i<24;i++))
do
path_source="/search/odin/guobk/data/dabaigou/00$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$i 2>&1 
done
i=24
path_source="/search/odin/guobk/data/dabaigou/00$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$i 2>&1 &
########################
gpu=7
for((i=25;i<29;i++))
do
path_source="/search/odin/guobk/data/dabaigou/00$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$i 2>&1 
done
i=29
path_source="/search/odin/guobk/data/dabaigou/00$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$i 2>&1 &
########################
gpu=1
for((i=30;i<33;i++))
do
path_source="/search/odin/guobk/data/dabaigou/00$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$i 2>&1 
done
i=33
path_source="/search/odin/guobk/data/dabaigou/00$i/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$i.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$i 2>&1 &
########################