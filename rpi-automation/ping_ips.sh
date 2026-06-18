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
  echo "[$name] -> $ip"
  ping -c 1 "$ip"
  ssh -n pi@"$ip" "echo $(hostname) ok"
done < "$ips_fn"