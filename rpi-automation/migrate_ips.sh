#!/usr/bin/env bash
set -euo pipefail

# Shell script that migrates IP addresses for Raspberry Pis from one WiFi network to another
# 
# Usage: ./migrate_ips.sh <SUBNET> <KNOWN_FILE> <MIGRATION_TXT>
# Example: ./migrate_ips.sh 128.174.61.128/26 dprgnet_ips.txt illinoisnet_ips.txt
#
# KNOWN_FILE format (will be created if missing): device new_ip
# MIGRATION_TXT format: device old_ip   (header allowed)
SUBNET="${1:-128.174.61.128/26}"
KNOWN_FILE="${2:-dprgnet_ips.txt}"
MIGRATE_FILE="${3:-illinoisnet_ips.txt}"
NEW_NETWORK_SSID="${4:-NETGEAR75}"
NEW_NETWORK_PASSWORD="${5:-***REMOVED***}"
NMAP_BIN="${NMAP_BIN:-nmap}"                 # path to nmap if not on PATH
SLEEP_AFTER_SWITCH="${SLEEP_AFTER_SWITCH:-10}"   # seconds to wait for WiFi network switch and DHCP

# --- helpers -----------------------------------------------------------------

require() { command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1"; exit 1; }; }
require "$NMAP_BIN"

# ensure known file exists with header
if [[ ! -f "$KNOWN_FILE" ]]; then
  echo "# name ip" > "$KNOWN_FILE"
fi

# read known IPs into a set
declare -A KNOWN_IP
if [[ -s "$KNOWN_FILE" ]]; then
  while read -r dev ip; do
    [[ -z "${ip// }" || "$dev" =~ ^# ]] && continue
    KNOWN_IP["$ip"]=1
  done < <(tail -n +2 "$KNOWN_FILE")
fi

# scan subnet, return only IPs (one per line)
scan_ips() {
  # -sn = host discovery only, -oG = greppable for easy awk parsing
  "$NMAP_BIN" -sn -oG - "$SUBNET" 2>/dev/null | \
    awk '/Status: Up/ {print $2}' | sort -V
}

# read migration file into arrays (device[i], oldip[i])
declare -a DEVICES OLDIPS
# handle both "device old_ip" and "device,old_ip" formats
while read -r dev old; do
  # skip empties and header-ish lines
  [[ -z "${dev// }" || "$dev" =~ ^# ]] && continue
  if [[ "${dev,,}" == "device" || "${dev,,}" == "name" ]]; then
    continue
  fi
  DEVICES+=("$dev")
  OLDIPS+=("$old")
done < "$MIGRATE_FILE"

if (( ${#DEVICES[@]} == 0 )); then
  echo "No devices found in $MIGRATE_FILE (expecting format: device ip)."
  exit 1
fi

echo "Target subnet: $SUBNET"
echo "Known file   : $KNOWN_FILE"
echo "Migration    : $MIGRATE_FILE  (${#DEVICES[@]} devices)"
echo

echo "Taking initial snapshot of active IPs on $SUBNET ..."
mapfile -t BASELINE < <(scan_ips)
echo "Baseline has ${#BASELINE[@]} IP(s): ${BASELINE[*]}."

# Build a quick membership set for baseline
declare -A BASESET
for ip in "${BASELINE[@]}"; do BASESET["$ip"]=1; done

# main loop over migration devices (one-by-one plug)
for i in "${!DEVICES[@]}"; do
  dev="${DEVICES[$i]}"
  old="${OLDIPS[$i]}"

  echo
  echo "================================================================"
  echo " Next device: $dev   (old IP: ${old:-unknown})"
  echo " Connecting $dev to WiFi network: $NEW_NETWORK_SSID"
  
  # Connect to the Pi using old IP and configure new WiFi
  if [[ -n "$old" && "$old" != "unknown" ]]; then
    echo " Configuring WiFi on $dev at $old..."
    
    # Connect to new WiFi network using NetworkManager
    echo " Connecting to $NEW_NETWORK_SSID..."
    echo " Note: SSH connection will be lost when WiFi switches networks."
    
    # Create a simple script in home directory and run it in background
    timeout 15 ssh -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -n pi@"$old" "
      echo '#!/bin/bash' > ~/wifi_connect.sh
      echo 'sudo nmcli dev wifi rescan' >> ~/wifi_connect.sh
      echo 'sleep 5' >> ~/wifi_connect.sh
      echo 'sudo nmcli dev wifi connect \"$NEW_NETWORK_SSID\" password \"$NEW_NETWORK_PASSWORD\"' >> ~/wifi_connect.sh
      chmod +x ~/wifi_connect.sh
      nohup ~/wifi_connect.sh &
    " || {
      echo " SSH timeout or WiFi connection script completed (this is normal when WiFi switches)."
    }
  else
    echo " No old IP available. Please connect $dev to $NEW_NETWORK_SSID manually."
    read -r -p " Press Enter when connected (or 'q' to quit) " ans || true
    [[ "${ans,,}" == "q" ]] && { echo "Aborted."; exit 0; }
  fi

  echo " Waiting ${SLEEP_AFTER_SWITCH}s for WiFi connection + DHCP..."
  sleep "$SLEEP_AFTER_SWITCH"

  echo " Scanning for newly appeared host ..."
  mapfile -t NOW < <(scan_ips)

  # compute NEW = NOW - BASELINE - KNOWN_IP
  declare -a CANDIDATES=()
  for ip in "${NOW[@]}"; do
    [[ -n "${BASESET[$ip]:-}" ]] && continue
    [[ -n "${KNOWN_IP[$ip]:-}" ]] && continue
    CANDIDATES+=("$ip")
  done

  if (( ${#CANDIDATES[@]} == 0 )); then
    echo " !! No new IPs detected. WiFi connection may still be in progress."
    read -r -p " Retry scan now? [Y/n] " retry < /dev/tty
    if [[ "${retry,,}" != "n" ]]; then
      ((i--))  # redo this device
      continue
    else
      echo " Skipping $dev."
      continue
    fi
  fi

  chosen_ip=""
  if (( ${#CANDIDATES[@]} == 1 )); then
    chosen_ip="${CANDIDATES[0]}"
    echo " Found new IP: $chosen_ip"
  else
    echo " Multiple new IPs detected:"
    for j in "${!CANDIDATES[@]}"; do
      printf "  [%d] %s\n" "$((j+1))" "${CANDIDATES[$j]}"
    done
    read -r -p " Select the index for $dev: " pick
    if ! [[ "$pick" =~ ^[0-9]+$ ]] || (( pick < 1 || pick > ${#CANDIDATES[@]} )); then
      echo " Invalid selection; skipping $dev."
      continue
    fi
    chosen_ip="${CANDIDATES[$((pick-1))]}"
  fi

  # append to known file: device ip (space-separated to match ethernet_ips.txt format)
  echo "${dev} ${chosen_ip}" >> "$KNOWN_FILE"
  KNOWN_IP["$chosen_ip"]=1
  BASESET["${chosen_ip}"]=1
  echo " Added: ${dev} ${chosen_ip}  ->  ${KNOWN_FILE}"
done

echo
echo "All done. Known IPs saved in: $KNOWN_FILE"
