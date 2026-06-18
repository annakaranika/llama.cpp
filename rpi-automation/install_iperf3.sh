#!/usr/bin/env bash
# install_iperf3.sh
# Install iperf3 on all active RPis listed in an IP file.
# Usage: ./install_iperf3.sh [ip_list_file]
#   ip_list_file - file with "name ip" lines (default: dprgnet_ips.txt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPS_FILE="${1:-${SCRIPT_DIR}/dprgnet_ips.txt}"

if [[ ! -f "$IPS_FILE" ]]; then
  echo "Error: file not found: $IPS_FILE" >&2; exit 1
fi

ok=0; fail=0

while read -r name ip; do
  [[ -z "$name" || "$name" =~ ^# ]] && continue

  printf "%-12s %s ... " "$name" "$ip"
  if ssh -n -o ConnectTimeout=5 -o StrictHostKeyChecking=no pi@"$ip" \
      "sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq iperf3 && sudo systemctl enable --now iperf3" 2>&1 | tail -1; then
    echo "✓ done"
    ((ok++)) || true
  else
    echo "✗ FAILED"
    ((fail++)) || true
  fi
done < "$IPS_FILE"

echo
echo "Done: $ok succeeded, $fail failed."
