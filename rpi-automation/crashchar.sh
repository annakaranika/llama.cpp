#!/bin/bash
# Pin the multi-ubatch prefill crash. Runs ENTIRELY on rpi1 (orchestrates the 4 Pis), so it survives
# the Mac logging off. Launch detached:
#   rsync rpi-automation/crashchar.sh rpi1-eth:/tmp/ ; \
#   ssh rpi1-eth "cd ~/llama.cpp && setsid nohup bash /tmp/crashchar.sh >/tmp/crashchar.txt 2>&1 </dev/null &"
# Then read /tmp/crashchar.txt on rpi1 whenever you're back online.
#
# Matrix isolates WHICH layer crashes: baseline (everything off) tells us if it's a pre-existing TP
# bug or an opt regression; per-opt-off tells us which opt; the server log gives the real crash site.
set -u
P="192.168.4.24 192.168.4.25 192.168.4.20 192.168.4.22"
RPC=192.168.4.24:50052,192.168.4.25:50052,192.168.4.20:50052,192.168.4.22:50052
M=~/llama.cpp/models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
SSH="ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=no"

# prompts that span 1, 2, and 3 ubatches (n_ubatch=512): ~300, ~700, ~1100 tokens
python3 -c "print('The history of ancient Greece spans many centuries and many city states. ' * 25)" > /tmp/p1.txt
python3 -c "print('The history of ancient Greece spans many centuries and many city states. ' * 60)" > /tmp/p2.txt
python3 -c "print('The history of ancient Greece spans many centuries and many city states. ' * 95)" > /tmp/p3.txt
echo "prompt sizes (words): $(wc -w </tmp/p1.txt) $(wc -w </tmp/p2.txt) $(wc -w </tmp/p3.txt)"

restart() { # $1 = server env (applied to all servers)
  for ip in $P; do
    $SSH pi@$ip "pkill -9 -x rpc-server 2>/dev/null; while pgrep -x rpc-server>/dev/null;do sleep 0.3;done; cd ~/llama.cpp && (setsid env $1 build-rpc/bin/rpc-server -H $ip -p 50052 -m 2000 </dev/null >/tmp/rpc.log 2>&1 &)"
  done
  sleep 3
  for ip in $P; do
    ok=$($SSH pi@$ip "for i in \$(seq 1 40); do bash -c '(exec 3<>/dev/tcp/$ip/50052) 2>/dev/null' && { echo Y; break; }; sleep 0.5; done" | grep -c Y)
    [ "$ok" = "1" ] || echo "  ! $ip not listening"
  done
}
trial() { # $1=client_env $2=server_env $3=prompt $4=label
  restart "$2"
  cd ~/llama.cpp
  env $1 build-rpc/bin/llama-cli -m $M --rpc $RPC -ngl 23 -sm row -no-cnv -f $3 -n 2 --temp 0 >/tmp/t.out 2>/tmp/t.err
  if grep -q GGML_ASSERT /tmp/t.err; then
    echo "[$4] CRASH @ $(grep GGML_ASSERT /tmp/t.err | grep -oE 'ggml-rpc.cpp:[0-9]+' | head -1)"
    echo "      server(.24) log tail: $($SSH pi@192.168.4.24 'tail -3 /tmp/rpc.log' | tr '\n' '|')"
  elif grep -q 'prompt eval time' /tmp/t.err; then
    echo "[$4] OK   $(grep 'prompt eval time' /tmp/t.err | grep -oE '[0-9]+ tokens.*second')"
  else
    echo "[$4] ???  $(tail -1 /tmp/t.err)"
  fi
}

echo "=== which prompt length first crashes (full optimized) ==="
trial "" "" /tmp/p1.txt "opt 1ubatch(~300)"
trial "" "" /tmp/p2.txt "opt 2ubatch(~700)"
trial "" "" /tmp/p3.txt "opt 3ubatch(~1100)"
echo ""
echo "=== isolate the layer on the 2-ubatch prompt (~700) ==="
trial "RPC_NO_OPT=1"            "RPC_NO_OPT=1"            /tmp/p2.txt "baseline (all off)"
trial "RPC_NO_PERSIST_BUFFERS=1" ""                      /tmp/p2.txt "no-persist"
trial "RPC_NO_PREFETCH=1"       "RPC_NO_PREFETCH=1"      /tmp/p2.txt "no-prefetch"
trial "RPC_NO_GRAPH_ONEWAY=1"   "RPC_NO_GRAPH_ONEWAY=1"  /tmp/p2.txt "no-oneway"

for ip in $P; do $SSH pi@$ip "pkill -9 -x rpc-server 2>/dev/null"; done
echo "=== done ==="
