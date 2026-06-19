#!/bin/bash
# ibss.sh -- bring a Raspberry Pi (Debian 12 bookworm, onboard brcmfmac wifi)
# onto an open 802.11 IBSS (ad-hoc) cell, or revert it to infrastructure wifi.
#
# WHAT ACTUALLY WORKS (learned the hard way, 2026-06-18):
#   * The Pi onboard Broadcom wifi (brcmfmac) DOES support IBSS -- but ONLY via
#     the kernel's `iw ibss join`, NOT via NetworkManager/wpa_supplicant. nmcli
#     `802-11-wireless.mode adhoc` fails on an OPEN ad-hoc net with:
#         "Connection activation failed: 802.1X supplicant took too long to
#          authenticate"
#     So we take wlan0 away from NetworkManager and drive IBSS with `iw` directly
#     (this is effectively what the old buster `adhoc.sh` did via wpa_supplicant
#     -Dwext; on bookworm the clean equivalent is direct `iw`).
#   * `iw` is in /usr/sbin, which is NOT in the default non-login PATH -- a bare
#     `iw list` over ssh can come back empty and look like "IBSS unsupported".
#     It is supported; run as root / full path. `sudo iw list` shows `* IBSS`.
#   * Two brcmfmac nodes on the cell DO exchange data (verified: rpi1<->rpi25
#     ping 0% loss ~8ms) even though `iw dev wlan0 station dump` stays empty
#     (a brcmfmac IBSS reporting quirk -- trust the ping, not the dump).
#
# SAFETY: a Pi whose only uplink is wifi goes unreachable the instant wlan0
#   leaves the infra AP. `up` schedules a reboot backstop (default 30 min)
#   BEFORE switching; the IBSS config is NOT persisted, so if the switch strands
#   the Pi it reboots back to infra on its own. Cancel it once reachable:
#       sudo shutdown -c
#   A wired Pi (the coordinator rpi1) stays reachable over eth regardless --
#   pass a backstop of 0 to skip the reboot.
#
# CELL: SSID=IBSS-RPiNet, channel 1 (2412 MHz), open, 192.168.4.0/24
#   (mask 255.255.255.0). Convention: a Pi's cell IP is 192.168.4.<rpi-number>
#   (see rpi-automation/adhoc_ips.txt). rpi1 = 192.168.4.1 = wired anchor/jump.
#
# USAGE (run ON the Pi, needs sudo):
#   sudo ./ibss.sh up 192.168.4.25 [backstop_min]   # join the cell as .25
#   sudo ./ibss.sh down                             # revert to infra wifi
#   sudo ./ibss.sh status                           # show wlan0 mode + IP
#
# Reach a cell Pi from a laptop through the wired anchor rpi1 (eth):
#   ssh -J pi@<rpi1-eth-ip> pi@192.168.4.25
# (modern macOS can't join IBSS itself -- it dropped ad-hoc support -- so go
#  through rpi1 either as an ssh -J jump host or by adding a route:
#   sudo route -n add 192.168.4.0/24 <rpi1-eth-ip>   )
# ----------------------------------------------------------------------------
set -u
IW=/usr/sbin/iw
DEV=wlan0
SSID=IBSS-RPiNet
FREQ=2412          # channel 1 (2.4 GHz)
CIDR=24

infra_con() {
    nmcli -t -f NAME,DEVICE connection show --active 2>/dev/null \
        | awk -F: -v d="$DEV" '$2==d{print $1; exit}'
}

cmd_up() {
    local ip="${1:-}" backstop="${2:-30}"
    [ -z "$ip" ] && { echo "usage: ibss.sh up <ip> [backstop_min]" >&2; exit 2; }
    if [ "$backstop" -gt 0 ] 2>/dev/null; then
        echo "safety: rebooting in $backstop min unless cancelled (sudo shutdown -c)"
        shutdown -r +"$backstop"
    fi
    nmcli dev set "$DEV" managed no 2>/dev/null; sleep 1
    ip link set "$DEV" down
    $IW dev "$DEV" set type ibss
    ip link set "$DEV" up
    $IW dev "$DEV" ibss join "$SSID" "$FREQ" fixed-freq
    sleep 4
    ip addr flush dev "$DEV"
    ip addr add "$ip/$CIDR" dev "$DEV"
    echo "joined $SSID as $ip"
    $IW dev "$DEV" info | grep -E "ssid|type|channel"
    ip -br addr show "$DEV"
}

cmd_down() {
    local infra; infra="$(infra_con)"; [ -z "$infra" ] && infra=preconfigured
    $IW dev "$DEV" ibss leave 2>/dev/null
    ip addr flush dev "$DEV"
    ip link set "$DEV" down
    $IW dev "$DEV" set type managed
    ip link set "$DEV" up
    nmcli dev set "$DEV" managed yes
    nmcli con up "$infra" 2>/dev/null || nmcli dev connect "$DEV"
    shutdown -c 2>/dev/null
    echo "reverted to infra ($infra)"
    nmcli -t -f DEVICE,STATE,CONNECTION dev 2>/dev/null | grep "$DEV"
}

case "${1:-}" in
    up)     shift; cmd_up "$@" ;;
    down)   cmd_down ;;
    status) $IW dev "$DEV" info 2>/dev/null | grep -E "ssid|type|channel"; ip -br addr show "$DEV" ;;
    *)      echo "usage: ibss.sh {up <ip> [backstop_min] | down | status}" >&2; exit 2 ;;
esac
