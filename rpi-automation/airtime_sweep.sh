#!/bin/bash
# Airtime-contention sweep: k concurrent iperf3 senders -> one sink over the shared
# wireless cell. Aggregate throughput vs k gives the contention exponent directly:
#   aggregate(k) = k^(1-exp) * C1   =>   exp = 1 - log(agg(k)/C1)/log(k)
# exp=0 : no contention (parallel links).  exp=1 : fully serialized channel.
set -u
SINK=192.168.4.24
SENDERS=(192.168.4.25 192.168.4.20 192.168.4.22)
PORTS=(5201 5202 5203)
DUR=10
OUT=/tmp/airtime_run
rm -rf "$OUT"; mkdir -p "$OUT"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=12"

echo "starting iperf3 servers on sink $SINK ..."
$SSH pi@$SINK "pkill -x iperf3 2>/dev/null; sleep 1; for p in ${PORTS[*]}; do iperf3 -s -p \$p -D; done; sleep 1; echo servers_up=\$(pgrep -x iperf3 | wc -l)"

mbps() { grep -i receiver "$1" | grep -oiE "[0-9.]+ Mbits/sec" | grep -oE "[0-9.]+" | tail -1; }

for k in 1 2 3; do
  echo "=== k=$k concurrent sender(s) ==="
  i=0
  while [ $i -lt $k ]; do
    $SSH pi@${SENDERS[$i]} "iperf3 -c $SINK -p ${PORTS[$i]} -t $DUR -O 2 -f m" > "$OUT/k${k}_f${i}.txt" 2>&1 &
    i=$((i+1))
  done
  wait
  agg=0; i=0
  while [ $i -lt $k ]; do
    m=$(mbps "$OUT/k${k}_f${i}.txt"); m=${m:-0}
    printf "  flow %d  %s -> :%s   %s Mbps\n" "$i" "${SENDERS[$i]}" "${PORTS[$i]}" "$m"
    agg=$(awk "BEGIN{print $agg + $m}")
    i=$((i+1))
  done
  printf "  AGGREGATE k=%d : %s Mbps\n" "$k" "$agg"
  echo "$k $agg" >> "$OUT/agg.txt"
  sleep 2
done

$SSH pi@$SINK "pkill -x iperf3 2>/dev/null" >/dev/null 2>&1

echo
echo "=== summary: k  aggregate_Mbps  measured_exp ==="
awk 'NR==1{c1=$2} {agg[$1]=$2} END{
  print "  C1 (k=1) =", c1, "Mbps"
  for(k=2;k<=3;k++){ if(agg[k]>0 && c1>0){ e=1-log(agg[k]/c1)/log(k); printf "  k=%d agg=%.1f Mbps  ->  exp=%.2f\n",k,agg[k],e } }
}' "$OUT/agg.txt"

# ── Measured on the dprgnet IBSS cell, 2026-07-01 (sink rpi24; senders rpi25/20/22) ──
#   rep | k=1   k=2   k=3   (aggregate Mbps)     measured exp
#     1 | 54.4  50.3  52.0                       ~1.05
#     2 | 56.9  45.2  41.0                       ~1.30
# Aggregate is FLAT-to-DECLINING with concurrent senders → the channel is fully
# (to super-) serialized: exp ≈ 1.0–1.3. Independently confirms the ~0.85 fit from
# the -sm row end-to-end sweep (graph_partitioning/tests/root/calibrate_airtime.py).
# Recommended sim value: Network.airtime_contention_exp ≈ 0.85 (reproduces the
# workload through the ring model; the raw channel is even more serialized).
