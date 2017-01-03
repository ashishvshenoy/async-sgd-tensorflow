#!/bin/bash

export TF_LOG_DIR="/home/ubuntu/tf/logs"
source tfdefs.sh

# startserver.py has the specifications for the cluster.
start_cluster startserver.py

START=$(date +%s)
nohup python asyncsgd.py --task_index=0 > asynclog-0.out 2>&1&
sleep 2 # wait for variable to be initialized
nohup python asyncsgd.py --task_index=1 > asynclog-1.out 2>&1&
nohup python asyncsgd.py --task_index=2 > asynclog-2.out 2>&1&
nohup python asyncsgd.py --task_index=3 > asynclog-3.out 2>&1&
nohup python asyncsgd.py --task_index=4 > asynclog-4.out 2>&1&
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Program run time : $DIFF"
