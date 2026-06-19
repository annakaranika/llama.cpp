#!/bin/bash
# ibss_cluster.sh -- from the laptop, bring the RPi cluster onto the IBSS cell
# (or revert it), using rpi-automation/ibss.sh on each Pi. See ibss.sh for the
# why/how (bookworm + brcmfmac: IBSS only via direct `iw`, not NetworkManager).
#
# Model: rpi1 is the WIRED anchor/jump (eth stays up, no reboot backstop); the
# wifi-only peers join the cell with a reboot backstop and are reached THROUGH
# rpi1. A peer's cell IP is 192.168.4.<rpi-number> (derived from its name in
# peer_ips.txt); rpi1 = 192.168.4.1.
#
# Usage (from the repo root or rpi-automation/):
#   ./ibss_cluster.sh up        # anchor + peers onto the cell, verify each via jump
#   ./ibss_cluster.sh down      # revert every node to infra wifi
#   ./ibss_cluster.sh status    # show each node's wlan0 mode/IP
#   ./ibss_cluster.sh ssh 24    # ssh into 192.168.4.24 through the rpi1 jump
# ----------------------------------------------------------------------------
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANCHOR_CAMPUS=128.174.61.159    # rpi1 eth (campus-reachable, wired)
ANCHOR_CELL=192.168.4.1
PEERS_FILE="$SCRIPT_DIR/peer_ips.txt"
BACKSTOP=30                     # minutes; peer auto-reboots to infra if stranded
SSHO="-o ConnectTimeout=8 -o StrictHostKeyChecking=no"
# NOTE: every `while read ... done 3< file` loop below reads the peer list on fd 3,
# NOT stdin -- otherwise the ssh calls inside the loop swallow the file and only the
# first iteration runs (the classic "ssh eats the loop's stdin" trap).
JUMP() { ssh -J pi@"$ANCHOR_CAMPUS" $SSHO -o UserKnownHostsFile=/dev/null "$@"; }
cell_ip() { echo "192.168.4.$(echo "$1" | tr -dc 0-9)"; }   # rpi24 -> 192.168.4.24
push()    { scp -q $SSHO "$SCRIPT_DIR/ibss.sh" pi@"$1":/home/pi/ibss.sh; }

case "${1:-}" in
  up)
    echo "== anchor rpi1 ($ANCHOR_CAMPUS) -> cell $ANCHOR_CELL (no backstop, eth-safe) =="
    push "$ANCHOR_CAMPUS"; ssh pi@"$ANCHOR_CAMPUS" $SSHO "sudo bash ~/ibss.sh up $ANCHOR_CELL 0"
    while read -r name ip <&3; do
      [[ -z "$name" || "${name:0:1}" == "#" || -z "$ip" ]] && continue
      cip=$(cell_ip "$name"); echo "== $name $ip -> cell $cip (backstop ${BACKSTOP}m) =="
      push "$ip"
      ssh -n pi@"$ip" $SSHO "sudo nohup setsid bash ~/ibss.sh up $cip $BACKSTOP >/home/pi/ibss.log 2>&1 & echo launched"
    done 3< "$PEERS_FILE"
    echo "settling 20s..."; sleep 20
    echo "== verify each peer via rpi1 jump =="
    while read -r name ip <&3; do
      [[ -z "$name" || "${name:0:1}" == "#" || -z "$ip" ]] && continue
      cip=$(cell_ip "$name")
      r=$(JUMP -o ConnectTimeout=12 pi@"$cip" 'hostname' 2>/dev/null)
      [ -n "$r" ] && echo "  $name $cip: OK -> ssh -J pi@$ANCHOR_CAMPUS pi@$cip" \
                  || echo "  $name $cip: NOT REACHABLE (its ${BACKSTOP}m backstop will recover it)"
    done 3< "$PEERS_FILE"
    echo "NOTE: once happy, cancel each peer's backstop: ssh -J ... pi@<cell-ip> 'sudo shutdown -c'"
    ;;
  down)
    while read -r name ip <&3; do
      [[ -z "$name" || "${name:0:1}" == "#" || -z "$ip" ]] && continue
      cip=$(cell_ip "$name")
      # revert over the cell (detached: the ssh rides the cell it tears down), fall back to campus
      JUMP -o ConnectTimeout=8 pi@"$cip" 'sudo nohup setsid bash ~/ibss.sh down >/dev/null 2>&1 &' 2>/dev/null \
        || ssh -n pi@"$ip" $SSHO 'sudo bash ~/ibss.sh down' 2>/dev/null
      echo "  $name: down requested"
    done 3< "$PEERS_FILE"
    ssh pi@"$ANCHOR_CAMPUS" $SSHO 'sudo bash ~/ibss.sh down'
    ;;
  status)
    echo "[rpi1 anchor $ANCHOR_CELL]"; ssh pi@"$ANCHOR_CAMPUS" $SSHO 'sudo bash ~/ibss.sh status' 2>/dev/null
    while read -r name ip <&3; do
      [[ -z "$name" || "${name:0:1}" == "#" || -z "$ip" ]] && continue
      cip=$(cell_ip "$name"); echo "[$name $cip]"
      JUMP -o ConnectTimeout=8 pi@"$cip" 'sudo bash ~/ibss.sh status' 2>/dev/null || echo "  (not reachable on cell)"
    done 3< "$PEERS_FILE"
    ;;
  ssh)
    exec ssh -J pi@"$ANCHOR_CAMPUS" $SSHO -o UserKnownHostsFile=/dev/null pi@"192.168.4.${2:-1}"
    ;;
  *) echo "usage: ibss_cluster.sh {up|down|status|ssh <cell-num>}" >&2; exit 2 ;;
esac
