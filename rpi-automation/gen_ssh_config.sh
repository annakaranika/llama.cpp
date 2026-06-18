#!/usr/bin/env bash
# gen_ssh_config.sh
# Generate ~/.ssh/config Host entries from a name/IP file.
# Usage: ./gen_ssh_config.sh [ips_file] [user]
#   ips_file  - file with "name ip" lines, comments with # are skipped (default: dprgnet_ips.txt)
#   user      - remote SSH user (default: pi)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPS_FILE="${1:-${SCRIPT_DIR}/dprgnet_ips.txt}"
REMOTE_USER="${2:-pi}"
SSH_CONFIG="$HOME/.ssh/config"
MARKER="# BEGIN rpi-aliases"
END_MARKER="# END rpi-aliases"

if [[ ! -f "$IPS_FILE" ]]; then
  echo "Error: file not found: $IPS_FILE" >&2; exit 1
fi

# Build the new block
BLOCK="$MARKER"$'\n'
while IFS= read -r line; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  name=$(awk '{print $1}' <<< "$line")
  ip=$(awk '{print $2}' <<< "$line")
  [[ -z "$ip" ]] && continue
  BLOCK+=$(printf "Host %s\n    HostName %s\n    User %s\n    IdentityFile ~/.ssh/id_ed25519\n    StrictHostKeyChecking no\n\n\n" "$name" "$ip" "$REMOTE_USER")
done < "$IPS_FILE"
BLOCK+="$END_MARKER"

# Remove existing block if present, then append new one
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

# Strip old block between markers
TMP=$(mktemp)
awk "/$MARKER/{found=1} !found{print} /$END_MARKER/{found=0}" "$SSH_CONFIG" > "$TMP"
# Remove trailing blank lines left by removal
sed -i '' -e '/^$/N;/^\n$/d' "$TMP" 2>/dev/null || true

{ cat "$TMP"; echo ""; echo "$BLOCK"; } > "$SSH_CONFIG"
rm -f "$TMP"

echo "Written to $SSH_CONFIG. Aliases added:"
grep "^Host " <<< "$BLOCK"
