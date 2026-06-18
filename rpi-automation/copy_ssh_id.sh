#!/usr/bin/env bash
# copy_ssh_id.sh
# Copy SSH public key to all active RPis listed in dprgnet_ips.txt
# Usage: ./copy_ssh_id.sh [user] [identity_file]
#   user           - remote username (default: pi)
#   identity_file  - path to public key (default: ~/.ssh/id_ed25519.pub)
#
# If the SSH key pair doesn't exist yet at the identity file's path, the
# script generates a fresh ed25519 keypair non-interactively (no passphrase).
#
# Requires `sshpass`. If missing, the script tries to install it for you
# based on the host OS (apt on Debian/Raspbian/Ubuntu, dnf on RHEL/Fedora,
# pacman on Arch, brew tap on macOS). Falls back to a manual instruction
# if the OS or package manager isn't recognised.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPS_FILE="${SCRIPT_DIR}/dprgnet_ips.txt"
REMOTE_USER="${1:-pi}"
IDENTITY="${2:-$HOME/.ssh/id_ed25519.pub}"

# All non-loopback IPv4 addresses of this host, best-effort across Linux and
# macOS. Used to skip ssh-copy-id-to-self (which either hangs or fails).
get_local_ips() {
  {
    hostname -I 2>/dev/null | tr ' ' '\n'                              # Linux
    ip -4 -o addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1     # Linux (ip)
    ifconfig 2>/dev/null | awk '/inet /{print $2}'                      # macOS / BSD
  } | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' \
    | grep -v '^127\.' \
    | sort -u
}

# Returns 0 if $1 matches any address in the LOCAL_IPS array.
is_local_ip() {
  local target="$1" ip
  for ip in "${LOCAL_IPS[@]}"; do
    [[ "$ip" == "$target" ]] && return 0
  done
  return 1
}

ensure_ssh_key() {
  local pub="$1"
  local priv="${pub%.pub}"   # strip .pub to get the private-key path
  if [[ "$priv" == "$pub" ]]; then
    echo "Error: identity file must end in .pub (got '$pub')" >&2
    return 1
  fi
  if [[ -f "$pub" && -f "$priv" ]]; then
    return 0  # already there, nothing to do
  fi
  if [[ -f "$pub" || -f "$priv" ]]; then
    echo "Error: half-installed keypair at $priv / $pub — refusing to overwrite." >&2
    echo "       Delete both files and re-run, or pass a different path." >&2
    return 1
  fi

  echo "No SSH keypair at $priv — generating ed25519 keypair (no passphrase)..." >&2
  mkdir -p "$(dirname "$priv")"
  chmod 700 "$(dirname "$priv")"
  ssh-keygen -t ed25519 \
    -f "$priv" \
    -N "" \
    -C "rpi-cluster-$(whoami)@$(hostname -s 2>/dev/null || hostname)-$(date +%Y%m%d)" \
    >/dev/null
  echo "  → wrote $priv and $pub" >&2
}

ensure_sshpass() {
  command -v sshpass >/dev/null 2>&1 && return 0

  echo "sshpass not found — attempting install..." >&2

  # Detect OS / package manager
  if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew >/dev/null 2>&1; then
      echo "  → macOS detected; using Homebrew tap" >&2
      brew install hudochenkov/sshpass/sshpass
    else
      echo "  Homebrew not found. Install Homebrew first: https://brew.sh" >&2
      return 1
    fi
  elif command -v apt-get >/dev/null 2>&1; then
    echo "  → apt-get detected (Debian/Ubuntu/Raspbian)" >&2
    sudo apt-get update -qq && sudo apt-get install -y sshpass
  elif command -v dnf >/dev/null 2>&1; then
    echo "  → dnf detected (RHEL/Fedora/Rocky)" >&2
    sudo dnf install -y sshpass
  elif command -v yum >/dev/null 2>&1; then
    echo "  → yum detected (older RHEL/CentOS)" >&2
    sudo yum install -y sshpass
  elif command -v pacman >/dev/null 2>&1; then
    echo "  → pacman detected (Arch)" >&2
    sudo pacman -S --noconfirm sshpass
  elif command -v apk >/dev/null 2>&1; then
    echo "  → apk detected (Alpine)" >&2
    sudo apk add --no-cache sshpass
  else
    echo "  Could not detect a supported package manager." >&2
    echo "  Please install sshpass manually and re-run this script." >&2
    return 1
  fi

  # Verify it actually showed up
  command -v sshpass >/dev/null 2>&1
}

if [[ ! -f "$IPS_FILE" ]]; then
  echo "Error: IP list not found at $IPS_FILE" >&2; exit 1
fi
if ! ensure_ssh_key "$IDENTITY"; then
  exit 1
fi
if ! ensure_sshpass; then
  echo "Error: sshpass is still not available after install attempt." >&2
  exit 1
fi

# Detect local IPs so we can skip ssh-copy-id-to-self.
LOCAL_IPS=()
while IFS= read -r _ip; do LOCAL_IPS+=("$_ip"); done < <(get_local_ips)
if (( ${#LOCAL_IPS[@]} > 0 )); then
  echo "Local IPs detected: ${LOCAL_IPS[*]}"
fi

read -rsp "Password for ${REMOTE_USER}@<rpi>: " SSH_PASS
echo; echo

ok=0; fail=0; skipped=0

while IFS= read -r line; do
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  name=$(awk '{print $1}' <<< "$line")
  ip=$(awk '{print $2}' <<< "$line")
  [[ -z "$ip" ]] && continue

  printf "%-12s %s ... " "$name" "$ip"

  # If this IP belongs to the host running the script, install the public
  # key into our own authorized_keys directly — there's no remote SSH to do.
  if is_local_ip "$ip"; then
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    touch "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
    if grep -qxFf "$IDENTITY" "$HOME/.ssh/authorized_keys" 2>/dev/null; then
      echo "↻ self (key already in authorized_keys)"
    else
      cat "$IDENTITY" >> "$HOME/.ssh/authorized_keys"
      echo "✓ self (added to authorized_keys)"
    fi
    ((skipped++)) || true
    continue
  fi

  if sshpass -p "$SSH_PASS" ssh-copy-id \
      -i "$IDENTITY" \
      -o StrictHostKeyChecking=no \
      -o ConnectTimeout=5 \
      "${REMOTE_USER}@${ip}"; then
    echo "✓ ok"
    ((ok++)) || true
  else
    echo "✗ FAILED"
    ((fail++)) || true
  fi
done < "$IPS_FILE"

echo
echo "Done: $ok succeeded via SSH, $skipped handled locally, $fail failed."
