alias mhadoop='/search/odin/software/MarsJs/bin/hadoop'
hadoopclients=ml_research,3evrqV2R
mkdir log
mhadoop fs -Dhadoop.client.ugi=$hadoopclients -get VpaOutput_guobk/logdata-process/UserInputAll_seg/* ../../data/data_inputs/ >> ./log/log.getdata 2>&1 &