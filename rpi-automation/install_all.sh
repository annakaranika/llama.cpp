# Check if argument is provided
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ip_list_file>"
  echo "Example: $0 ips.txt"
  exit 1
fi

# Assign first argument to variable
ips_fn="$1"

# Verify file exists
if [[ ! -f "$ips_fn" ]]; then
  echo "Error: File '$ips_fn' not found"
  exit 1
fi

while read -r name ip; do
  [[ -z "$name" || "$name" =~ ^# ]] && continue
  echo "Installing at [$name] -> $ip"
  scp install_llamacpp.sh pi@"$ip":~/install_llamacpp.sh
  ssh -n pi@"$ip" "chmod +x ~/install_llamacpp.sh"
  # install tmux if not installed
  ssh -n pi@"$ip" "sudo apt update && sudo apt install -y tmux" > /dev/null 2>&1
  # create a tmux session and run install_llamacpp.sh inside it
  ssh -n pi@"$ip" "tmux new-session -d -s 'llama-install' '~/install_llamacpp.sh; exec bash'" &
done < "$ips_fn"

# repeatedly check that rpc server is built on each RPI
echo "Installation commands sent to all RPIs. Waiting for completion..."

all_done=false
start_time=$(date +%s)
while [ "$all_done" = false ]; do
  all_done=true
  while read -r name ip; do
    [[ -z "$name" || "$name" =~ ^# ]] && continue
    echo "Checking build status on [$name] -> $ip"
    status=$(ssh -n pi@"$ip" "if [ -f ~/llama.cpp/build-rpc/bin/rpc-server ]; then echo 'done'; else echo 'not done'; fi")
    if [ "$status" != "done" ]; then
        all_done=false
        # break while loop to wait for next check
        break
    fi
  done < "$ips_fn"
  if [ "$all_done" = false ]; then
    echo "Not all builds are complete. Checking again in 10 seconds..."
    sleep 10
  fi
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "All 'rpc-server' builds completed on all RPIs in $duration seconds."