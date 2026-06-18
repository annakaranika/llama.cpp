#!/bin/bash
# -----------------------------------------------------------------------------
# setup_mesh.sh
#
# Configures RPis in 802.11s WiFi mesh mode or restores infrastructure (AP) mode.
# Run this on EACH RPi individually via SSH, or use the --all flag to push to all.
#
# Usage (from laptop):
#   ./scripts/setup_mesh.sh mesh    # switch all RPis to 802.11s mesh
#   ./scripts/setup_mesh.sh ap      # restore all RPis to AP/infrastructure mode
#   ./scripts/setup_mesh.sh status  # show current mode on all RPis
#
# Requirements:
#   - iw, hostapd, wpasupplicant installed on RPis
#   - SSH key-based access
#   - RPis must share the same SSID/channel for mesh to form
#
# Notes:
#   - Mesh mode uses a dedicated mesh interface (mesh0) alongside wlan0
#   - The existing wlan0 (AP connection for SSH) is kept untouched so you
#     don't lose access. Inference traffic routes over mesh0.
#   - Pairwise mesh IPs are in the 10.10.10.x/24 subnet.
# -----------------------------------------------------------------------------
set -e

RPIS=(
    "172.16.107.154"
    "172.16.112.192"
    # Add more RPi IPs here
)
SSH_USER="pi"
MESH_SSID="llmiot_mesh"
MESH_CHANNEL=6
MESH_FREQ=2437   # channel 6 in MHz (2.4GHz). Use 5180 for 5GHz ch36 if supported.
MESH_SUBNET="10.10.10"
MODE="${1:-status}"

ssh_cmd() {
    local host="$1"; shift
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_USER}@${host}" "$@"
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Assign a stable mesh IP based on the last octet of the wlan0 IP
mesh_ip_for() {
    local rpi="$1"
    local octet="${rpi##*.}"
    echo "${MESH_SUBNET}.${octet}"
}

MESH_SETUP_SCRIPT='
set -e
SSID="'"$MESH_SSID"'"
FREQ='"$MESH_FREQ"'
MESH_IP="$1"

# Load mesh kernel module
sudo modprobe mac80211_mesh 2>/dev/null || true

# Remove old mesh interface if exists
sudo iw dev mesh0 del 2>/dev/null || true

# Get the physical wireless device name
PHY=$(iw dev wlan0 info | grep wiphy | awk "{print \$2}")

# Create a new mesh interface
sudo iw phy phy${PHY} interface add mesh0 type mesh

# Bring it up
sudo ip link set mesh0 up

# Join the mesh network
sudo iw dev mesh0 mesh join "$SSID" freq $FREQ

# Assign mesh IP
sudo ip addr flush dev mesh0 2>/dev/null || true
sudo ip addr add "${MESH_IP}/24" dev mesh0

echo "Mesh interface mesh0 up at ${MESH_IP}"
iw dev mesh0 info
'

AP_RESTORE_SCRIPT='
set -e
sudo iw dev mesh0 del 2>/dev/null || true
echo "Mesh interface removed, back to AP-only mode"
'

STATUS_SCRIPT='
echo "=== $(hostname) ==="
if iw dev mesh0 info 2>/dev/null; then
    echo "Mode: MESH"
    ip addr show mesh0 | grep "inet "
else
    echo "Mode: AP (no mesh0 interface)"
fi
ip addr show wlan0 | grep "inet " || true
'

case "$MODE" in
    mesh)
        log "Switching all RPis to 802.11s mesh mode..."
        IDX=1
        for rpi in "${RPIS[@]}"; do
            MESH_IP=$(mesh_ip_for "$rpi")
            log "  $rpi -> mesh IP $MESH_IP"
            ssh_cmd "$rpi" "MESH_IP=$MESH_IP; $MESH_SETUP_SCRIPT" "$MESH_IP" &
            ((IDX++))
        done
        wait
        log "All RPis in mesh mode. Test connectivity with:"
        for rpi in "${RPIS[@]}"; do
            echo "  ssh ${SSH_USER}@${rpi} 'ping -c 3 $(mesh_ip_for "${RPIS[0]}")'"
        done
        log "Then re-run measure_links.sh with MESH_MODE=1 to capture mesh link data."
        ;;
    ap)
        log "Restoring all RPis to AP/infrastructure mode..."
        for rpi in "${RPIS[@]}"; do
            ssh_cmd "$rpi" "$AP_RESTORE_SCRIPT" &
        done
        wait
        log "All RPis restored to AP mode."
        ;;
    status)
        for rpi in "${RPIS[@]}"; do
            ssh_cmd "$rpi" "$STATUS_SCRIPT"
            echo ""
        done
        ;;
    *)
        echo "Usage: $0 [mesh|ap|status]"
        exit 1
        ;;
esac
