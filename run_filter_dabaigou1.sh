gpu=$1
file1=$2
file2=$3
file3=$4
path_source="/search/odin/guobk/data/dabaigou/00$file1/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$file1.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$file1 2>&1
path_source="/search/odin/guobk/data/dabaigou/00$file2/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$file2.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$file2 2>&1
path_source="/search/odin/guobk/data/dabaigou/00$file3/part-00001"
path_target="/search/odin/guobk/data/dabaigou/filtered/00$file3.txt"
nohup python -u filter_dabaigou.py $path_source $path_target $gpu >> log/fiter-dabaigou-$file3 2>&1 &
