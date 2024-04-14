#!/bin/bash
# bthelmer Braden T Helmer

NODES=`echo $SLURM_NODELIST | tr -d c | tr -d [ | tr -d ] | perl -pe 's/(\d+)-(\d+)/join(",",$1..$2)/eg' | awk 'BEGIN { RS=","} { print "c"$1 }'`
NODES=($NODES)

node_len=0

WORKER_JSON="["
EVALUATOR_JSON="["
for NODE in "${NODES[@]}"; do
	NODE_WITH_PORT="$NODE:$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
	WORKER_JSON+="\"$NODE_WITH_PORT\","
	if [ $node_len -eq 0 ]; then
		EVALUATOR_JSON+="\"$NODE_WITH_PORT\"]"
	fi
	(( node_len++ ))
done
WORKER_JSON="${WORKER_JSON%,}]"
(( node_len-- ))

here=`pwd`

for (( i = node_len; i >= 0; i-- )); do
	NODE="${NODES[i]}"	
	TF_CONFIG_JSON="{
		\"cluster\": {
			\"worker\": $WORKER_JSON,
			\"evaluator\": $EVALUATOR_JSON
		},
		\"task\": {
			\"type\": \"worker\",
			\"index\": $i
		}
	}"
	if [ $i -eq 0 ]; then
				export TF_CONFIG=$TF_CONFIG_JSON; python3 cnnhw-par.py $i &> tmp0 &
				TF_CONFIG_JSON="{
					\"cluster\": {
						\"worker\": $WORKER_JSON,
						\"evaluator\": $EVALUATOR_JSON
					},
					\"task\": {
						\"type\": \"evaluator\",
						\"index\": $i
					}
				}"
				export TF_CONFIG=$TF_CONFIG_JSON; python3 cnnhw-par.py -1
	else
				ssh -n $NODE "cd $here; export TF_CONFIG='$TF_CONFIG_JSON'; python3 cnnhw-par.py $i &> tmp$i" &

	fi
done

