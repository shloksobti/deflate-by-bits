#!/usr/bin/env bash
# Wait for the GPU queue to finish (or a hard deadline), copy results, delete the droplet.
H=137.184.175.190; ID=605615728; DEADLINE=$(( $(date +%s) + 9000 ))
until ssh -o ConnectTimeout=10 root@$H 'grep -q ALLDONE /root/research/results/queue2.log && grep -q DONE_CODER5 /root/research/results/queue2.log' 2>/dev/null || [ $(date +%s) -gt $DEADLINE ]; do sleep 60; done
rsync -az root@$H:/root/research/results/ /Users/shloksobti/Desktop/personal/research/results/ && echo "synced"
doctl compute droplet delete $ID --force && echo "droplet $ID deleted at $(date)"
