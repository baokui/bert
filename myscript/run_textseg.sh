mkdir log/log_textseg
Date=20200104
for((Date=20200102;Date<=20200106;Date++))
done
for((i=0;i<24;i++))
do
if [ $i -gt 9 ]; then
Hour=$i
else
Hour=0$i
fi
sourcefile=../../data/data_inputs/$Date/$Hour/part-00001
b=$(( $i % 8 ))
if [ $b = 7 ]
then
nohup python -u textseg.py $sourcefile >> log/log_textseg/$Date-$Hour.log 2>&1
else
nohup python -u textseg.py $sourcefile >> log/log_textseg/$Date-$Hour.log 2>&1 &
fi
done
done