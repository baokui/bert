alias mhadoop='/search/odin/software/MarsJs/bin/hadoop'
hadoopclients=ml_research,3evrqV2R
mkdir log
mhadoop fs -Dhadoop.client.ugi=$hadoopclients -get VpaOutput_guobk/logdata-process/UserInputAll_seg/* ../../data/data_inputs/ >> ./log/log.getdata 2>&1 &

for((Date=20200103;Date<=20200106;Date++))
do
echo $Date
for((i=0;i<24;i++))
do
if [ $i -gt 9 ]; then
Hour=$i
else
Hour=0$i
fi
mkdir ../../data/data_inputs/$Date
mhadoop fs -Dhadoop.client.ugi=$hadoopclients -get VpaOutput_guobk/logdata-process/UserInputAll_seg/$Date/$Hour ../../data/data_inputs/$Date >> ./log/$Date-$Hour.log 2>&1 &
done
done

Date=20200102
for((i=12;i<24;i++))
do
if [ $i -gt 9 ]; then
Hour=$i
else
Hour=0$i
fi
mkdir ../../data/data_inputs/$Date
mhadoop fs -Dhadoop.client.ugi=$hadoopclients -get VpaOutput_guobk/logdata-process/UserInputAll_seg/$Date/$Hour ../../data/data_inputs/$Date >> ./log/$Date-$Hour.log 2>&1 &
done