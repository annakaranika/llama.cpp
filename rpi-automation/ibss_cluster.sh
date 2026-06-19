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
#   ./ibss_cluster.sh up         # anchor + peers onto the cell, verify each via jump
#   ./ibss_cluster.sh down       # revert every node to infra wifi
#   ./ibss_cluster.sh status     # show each node's wlan0 mode/IP
#   ./ibss_cluster.sh ssh 24     # ssh into 192.168.4.24 through the rpi1 jump
#   ./ibss_cluster.sh ssh-config # add a ProxyJump alias so `ssh pi@192.168.4.<n>` works
#                                # directly (the right tool when behind a router/VPN)
#   ./ibss_cluster.sh route del  # remove a stale Mac route (route add only works if the
#                                # Mac shares rpi1's L2 segment -- usually it doesn't)
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
  ssh-config)
    # The reliable way to get `ssh pi@192.168.4.<num>` from a laptop that reaches
    # rpi1 through a router/VPN: an ssh ProxyJump alias (NOT an IP route -- see
    # `route`). Idempotent; appends a marked block to ~/.ssh/config.
    CFG="$HOME/.ssh/config"; MARK="# >>> ibss cell via rpi1 jump >>>"
    if grep -qE "Host .*192\.168\.4" "$CFG" 2>/dev/null; then
      echo "a Host block matching 192.168.4.* already exists in $CFG -- leaving it"
    else
      mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"
      cat >> "$CFG" <<EOF

$MARK
Host 192.168.4.*
    ProxyJump pi@$ANCHOR_CAMPUS
    User pi
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
# <<< ibss cell <<<
EOF
      echo "added ProxyJump block to $CFG  ->  now: ssh pi@192.168.4.<num>"
    fi
    ;;
  route)
    # Direct Mac route into the cell via rpi1 -- ONLY works if this Mac is on the
    # SAME L2 segment as rpi1 (an IP route can't use an off-link gateway). Behind a
    # router/VPN it can't work; use `ssh-config` instead. `route del` tears down.
    sub="${2:-add}"
    if [ "$sub" = add ]; then
      gw=$(route -n get "$ANCHOR_CAMPUS" 2>/dev/null | awk '/gateway:/{print $2}')
      if [ -n "$gw" ] && [ "$gw" != "$ANCHOR_CAMPUS" ]; then
        echo "ABORT: this Mac reaches rpi1 ($ANCHOR_CAMPUS) via router $gw, not on-link."
        echo "       An IP route needs an on-link gateway, so it can't work over a"
        echo "       router/VPN here. Use the ssh ProxyJump instead:"
        echo "         ./ibss_cluster.sh ssh-config     # then: ssh pi@192.168.4.<num>"
        exit 1
      fi
      echo "1/3 rpi1 ip_forward=1..."
      ssh pi@"$ANCHOR_CAMPUS" $SSHO 'sudo sysctl -w net.ipv4.ip_forward=1 >/dev/null && echo ok'
      echo "2/3 Mac route 192.168.4.0/24 -> $ANCHOR_CAMPUS (sudo)..."
      sudo route -n add 192.168.4.0/24 "$ANCHOR_CAMPUS"
      echo "3/3 peer return routes (default via $ANCHOR_CELL)..."
      while read -r name ip <&3; do
        [[ -z "$name" || "${name:0:1}" == "#" || -z "$ip" ]] && continue
        cip=$(cell_ip "$name")
        JUMP -o ConnectTimeout=8 pi@"$cip" "sudo ip route replace default via $ANCHOR_CELL dev wlan0" 2>/dev/null \
          && echo "  $name ok" || echo "  $name unreachable"
      done 3< "$PEERS_FILE"
      echo "done -> now: ssh pi@192.168.4.<num>   (no -J needed)"
    else
      sudo route -n delete 192.168.4.0/24 2>/dev/null && echo "removed Mac route 192.168.4.0/24"
    fi
    ;;
  *) echo "usage: ibss_cluster.sh {up|down|status|ssh <cell-num>|ssh-config|route [add|del]}" >&2; exit 2 ;;
esac
