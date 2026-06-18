#!/bin/zsh
# -----------------------------------------------------------------------------
# measure_links.sh
#
# Measures pairwise bandwidth (iperf3) and RTT (ping) between RPi nodes.
#
# Usage:
#   ./rpi-automation/measure_links.sh [--csv <file>] [--ips <file>] [--parallel]
#                                      [--client-rtts] [--nodes n1 [n2 ...]]
#
# Modes (mutually exclusive; default is full measurement):
#   (default)       Measure all pairs (client<->rpi and rpi<->rpi). Overwrites CSV.
#   --client-rtts   Only ping client->rpi for rows with a missing RTT.
#                   Updates the existing CSV in place.
#   --pair a b      Measure both directions between exactly two nodes.
#                   Updates the existing CSV in place.
#   --nodes n1 n2   Measure all pairs involving these nodes only (e.g. when
#                   adding new nodes). Appends results to the existing CSV.
#
# Flags:
#   --csv  <file>   Output CSV (default: graph_partitioning/tests/pair_links_measured.csv)
#   --ips  <file>   IP list file, "name ip" lines (default: rpi-automation/dprgnet_ips.txt)
#   --parallel      Measure non-overlapping pairs simultaneously.
#
# Requirements:
#   - iperf3 on all RPis and this machine (brew install iperf3)
#   - SSH key-based access to all RPis
# -----------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Defaults ----
CSV="graph_partitioning/tests/pair_links_measured.csv"
IPS_FILE="${SCRIPT_DIR}/dprgnet_ips.txt"
PARALLEL=false
MODE="full"       # full | client-rtts | nodes | pair
NEW_NODES=()      # populated by --nodes
PAIR=()           # populated by --pair (exactly two names)

SSH_USER="pi"
IPERF_PORT=5201
IPERF_DURATION=5
PING_COUNT=20

# ---- Argument parsing ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)      CSV="$2";       shift 2 ;;
        --ips)      IPS_FILE="$2";  shift 2 ;;
        --parallel|-p) PARALLEL=true; shift ;;
        --client-rtts) MODE="client-rtts"; shift ;;
        --nodes)
            MODE="nodes"
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                NEW_NODES+=("$1"); shift
            done
            ;;
        --pair)
            MODE="pair"
            PAIR=("$2" "$3"); shift 3
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$IPS_FILE" ]]; then
    echo "ERROR: IP list not found: $IPS_FILE" >&2; exit 1
fi

# ---- Load IP list ----
RPI_NAMES=()
RPIS=()
while read -r _n _i; do
    [[ -z "$_n" || "${_n[1]}" == "#" || -z "$_i" ]] && continue
    RPI_NAMES+=("$_n")
    RPIS+=("$_i")
done < "$IPS_FILE"

log()     { echo "[$(date '+%H:%M:%S')] $*"; }
ssh_cmd() { local h="$1"; shift; ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${SSH_USER}@${h}" "$@"; }

# Print a human-readable duration estimate before starting measurements.
# $1 = number of full pairs (bidirectional iperf3 + ping)
# $2 = number of ping-only measurements (client-rtts mode)
print_estimate() {
    local pairs=$1 pings_only=$2
    # ~50s per full pair (2×iperf3 + 2×sleep + 2×ping), ~20s per ping-only
    local secs=$(( pairs * (2 * IPERF_DURATION + 2 + (PING_COUNT - 1)) + pings_only * PING_COUNT ))
    local h=$(( secs / 3600 ))
    local m=$(( (secs % 3600) / 60 ))
    local s=$(( secs % 60 ))
    if (( h > 0 )); then
        echo "Estimated time: ${h}h ${m}m ${s}s  (${pairs} pairs × ~$((2*IPERF_DURATION+2+2*(PING_COUNT-1)))s each)"
    else
        echo "Estimated time: ${m}m ${s}s  (${pairs} pairs × ~$((2*IPERF_DURATION+2+2*(PING_COUNT-1)))s each)"
    fi
}

# Client's own IP as seen on the network (used for rpi→client iperf3)
CLIENT_IP=$(python3 -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0])")

ip_of() {
    local _name="$1" _i
    for _i in {1..${#RPI_NAMES[@]}}; do
        [[ "${RPI_NAMES[$_i]}" == "$_name" ]] && { echo "${RPIS[$_i]}"; return; }
    done
    echo ""
}

# ---- Core measurement helpers ----

# ping_rtt <ip>  →  prints avg RTT in ms (or "" on failure)
ping_rtt() {
    ping -c "$PING_COUNT" -q "$1" 2>/dev/null \
        | grep -E "rtt min|round-trip" | awk -F'/' '{print $5}'
}

# extract_bw <json>  →  prints Mbps or "0"
extract_bw() {
    echo "$1" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(round(d['end']['sum_received']['bits_per_second']/1e6,2))" \
        2>/dev/null || echo "0"
}

# measure_pair <src_ip|"client"> <dst_ip> <src_name> <dst_name> <outfile>
# Measures both directions and appends two CSV rows to outfile.
measure_pair() {
    local src="$1" dst="$2" src_name="$3" dst_name="$4" outfile="$5"
    local BW_FWD BW_BWD RTT srv_pid

    if [[ "$src" == "client" ]]; then
        # RTT: ping once from client — symmetric, shared for both rows
        RTT=$(ping_rtt "$dst")

        # Forward: client → rpi
        ssh_cmd "$dst" "pkill iperf3 2>/dev/null || true; iperf3 -s -p $IPERF_PORT -D --one-off" 2>/dev/null
        sleep 1
        BW_FWD=$(extract_bw "$(iperf3 -c "$dst" -p "$IPERF_PORT" -t "$IPERF_DURATION" -J 2>/dev/null)")

        # Backward: rpi → client
        iperf3 -s -p "$IPERF_PORT" --one-off 2>/dev/null &
        srv_pid=$!
        sleep 1
        BW_BWD=$(extract_bw "$(ssh_cmd "$dst" "iperf3 -c $CLIENT_IP -p $IPERF_PORT -t $IPERF_DURATION -J 2>/dev/null")")
        wait "$srv_pid" 2>/dev/null || true
    else
        # RTT: ping once from src — symmetric, shared for both rows
        RTT=$(ssh_cmd "$src" \
            "ping -c $PING_COUNT -q $dst 2>/dev/null | grep 'rtt min' | awk -F'/' '{print \$5}'" || echo "")

        # Forward: rpi_src → rpi_dst
        ssh_cmd "$dst" "pkill iperf3 2>/dev/null || true; iperf3 -s -p $IPERF_PORT -D --one-off" 2>/dev/null
        sleep 1
        BW_FWD=$(extract_bw "$(ssh_cmd "$src" "iperf3 -c $dst -p $IPERF_PORT -t $IPERF_DURATION -J 2>/dev/null")")

        # Backward: rpi_dst → rpi_src
        ssh_cmd "$src" "pkill iperf3 2>/dev/null || true; iperf3 -s -p $IPERF_PORT -D --one-off" 2>/dev/null
        sleep 1
        BW_BWD=$(extract_bw "$(ssh_cmd "$dst" "iperf3 -c $src -p $IPERF_PORT -t $IPERF_DURATION -J 2>/dev/null")")
    fi

    echo "${src_name},${dst_name},${BW_FWD},${RTT}" >> "$outfile"
    echo "${dst_name},${src_name},${BW_BWD},${RTT}" >> "$outfile"
}

# merge_into_csv <new_rows_file> <csv_file>
# Appends/updates rows from new_rows_file into csv_file using python3.
merge_into_csv() {
    local new_rows="$1" target_csv="$2"
    python3 - "$new_rows" "$target_csv" <<'EOF'
import sys, csv, os

new_rows_path, csv_path = sys.argv[1], sys.argv[2]

new_rows = {}
with open(new_rows_path, newline="") as f:
    for row in csv.reader(f):
        if len(row) >= 3:
            new_rows[(row[0], row[1])] = row

if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["device1", "device2", "bw_mbps", "rtt_ms"])

existing = []
seen = set()
with open(csv_path, newline="") as f:
    for row in csv.reader(f):
        if not row:
            continue
        key = (row[0], row[1])
        if key in new_rows:
            existing.append(new_rows.pop(key))
        else:
            existing.append(row)
        seen.add(key)

# Append any pairs not already in the CSV
for row in new_rows.values():
    existing.append(row)

tmp = csv_path + ".tmp"
with open(tmp, "w", newline="") as f:
    csv.writer(f).writerows(existing)
os.replace(tmp, csv_path)
EOF
}

# ===========================================================================
# MODE: pair — measure both directions between exactly two nodes
# ===========================================================================
if [[ "$MODE" == "pair" ]]; then
    if [[ ${#PAIR[@]} -ne 2 ]]; then
        echo "ERROR: --pair requires exactly two node names, e.g. --pair rpi1 rpi2" >&2; exit 1
    fi
    a_name="${PAIR[1]}"; b_name="${PAIR[2]}"
    a_ip=$(ip_of "$a_name"); b_ip=$(ip_of "$b_name")

    # "client" is a valid node name — resolve its IP too
    [[ "$a_name" == "client" ]] && a_ip="client"
    [[ "$b_name" == "client" ]] && b_ip="client"

    if [[ -z "$a_ip" ]]; then echo "ERROR: no IP for $a_name" >&2; exit 1; fi
    if [[ -z "$b_ip" ]]; then echo "ERROR: no IP for $b_name" >&2; exit 1; fi

    TMPDIR_MEAS=$(mktemp -d)
    COLLECTED="$TMPDIR_MEAS/collected.csv"
    trap 'rm -rf "$TMPDIR_MEAS"' EXIT

    print_estimate 1 0
    log "Measuring $a_name <-> $b_name (both directions)..."
    measure_pair "$a_ip" "$b_ip" "$a_name" "$b_name" "$COLLECTED"
    log "  a→b: $(head -1 "$COLLECTED" | cut -d, -f3) Mbps, $(head -1 "$COLLECTED" | cut -d, -f4) ms"
    log "  b→a: $(tail -1 "$COLLECTED" | cut -d, -f3) Mbps, $(tail -1 "$COLLECTED" | cut -d, -f4) ms"

    merge_into_csv "$COLLECTED" "$CSV"
    log "Done. Updated $CSV"
    exit 0
fi

# ===========================================================================
# MODE: client-rtts — only fill missing client<->rpi RTTs via ping
# ===========================================================================
if [[ "$MODE" == "client-rtts" ]]; then
    if [[ ! -f "$CSV" ]]; then
        echo "ERROR: CSV not found: $CSV" >&2; exit 1
    fi

    # Find rpi names with blank client RTT
    missing=()
    while IFS=, read -r d1 d2 bw rtt; do
        [[ "$d1" == "device1" ]] && continue
        [[ "$d1" != "client" ]] && continue
        [[ -n "${rtt// /}" ]] && continue
        missing+=("$d2")
    done < "$CSV"

    if [[ ${#missing[@]} -eq 0 ]]; then
        echo "No missing client RTTs found in $CSV."; exit 0
    fi
    print_estimate 0 ${#missing[@]}
    log "Found ${#missing[@]} RPis with missing client RTT: ${missing[*]}"

    TMPRTTS=$(mktemp)
    trap 'rm -f "$TMPRTTS"' EXIT

    for rpi_name in "${missing[@]}"; do
        ip=$(ip_of "$rpi_name")
        if [[ -z "$ip" ]]; then
            log "WARNING: no IP for $rpi_name, skipping"; continue
        fi
        log "Pinging $rpi_name ($ip)..."
        rtt_val=$(ping_rtt "$ip")
        if [[ -z "$rtt_val" ]]; then
            log "WARNING: ping failed for $rpi_name, skipping"; continue
        fi
        log "  $rpi_name RTT = ${rtt_val} ms"
        echo "$rpi_name $rtt_val" >> "$TMPRTTS"
    done

    python3 - "$CSV" "$TMPRTTS" <<'EOF'
import sys, csv, os
csv_path, rtts_path = sys.argv[1], sys.argv[2]
rtts = {}
with open(rtts_path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            rtts[parts[0]] = parts[1]
rows = []
with open(csv_path, newline="") as f:
    for row in csv.reader(f):
        if len(row) == 4:
            d1, d2, bw, rtt = row
            if not rtt.strip():
                if d1 == "client" and d2 in rtts:
                    rtt = rtts[d2]
                elif d2 == "client" and d1 in rtts:
                    rtt = rtts[d1]
            rows.append([d1, d2, bw, rtt])
        else:
            rows.append(row)
tmp = csv_path + ".tmp"
with open(tmp, "w", newline="") as f:
    csv.writer(f).writerows(rows)
os.replace(tmp, csv_path)
print(f"Updated {len(rtts)} RTT(s) in {csv_path}.")
EOF
    log "Done."
    exit 0
fi

# ===========================================================================
# MODE: nodes — measure all pairs involving the specified new nodes
# ===========================================================================
if [[ "$MODE" == "nodes" ]]; then
    if [[ ${#NEW_NODES[@]} -eq 0 ]]; then
        echo "ERROR: --nodes requires at least one node name." >&2; exit 1
    fi

    TMPDIR_MEAS=$(mktemp -d)
    COLLECTED="$TMPDIR_MEAS/collected.csv"
    trap 'rm -rf "$TMPDIR_MEAS"' EXIT
    touch "$COLLECTED"

    N=${#RPI_NAMES[@]}
    k=${#NEW_NODES[@]}
    # pairs: client↔each new + existing↔each new + new↔new
    node_pairs=$(( k + (N - k) * k + k * (k - 1) / 2 ))
    print_estimate $node_pairs 0
    log "Measuring pairs involving: ${NEW_NODES[*]}"

    for new_name in "${NEW_NODES[@]}"; do
        new_ip=$(ip_of "$new_name")
        if [[ -z "$new_ip" ]]; then
            echo "ERROR: no IP found for $new_name in $IPS_FILE" >&2; exit 1
        fi

        # client <-> new node
        log "Measuring client <-> $new_name..."
        tmpf="$TMPDIR_MEAS/c_${new_name}.csv"
        measure_pair "client" "$new_ip" "client" "$new_name" "$tmpf"
        cat "$tmpf" >> "$COLLECTED"
        log "  -> $(head -1 "$tmpf" | cut -d, -f3) Mbps, $(head -1 "$tmpf" | cut -d, -f4) ms"

        # all existing nodes <-> new node
        for (( i=1; i<=N; i++ )); do
            existing_name="${RPI_NAMES[$i]}"
            existing_ip="${RPIS[$i]}"
            [[ "$existing_name" == "$new_name" ]] && continue
            # skip if also a new node (will be covered when we process the other new node)
            skip=false
            for nn in "${NEW_NODES[@]}"; do
                [[ "$existing_name" == "$nn" ]] && skip=true && break
            done
            $skip && continue

            log "Measuring $existing_name <-> $new_name..."
            tmpf="$TMPDIR_MEAS/${existing_name}_${new_name}.csv"
            measure_pair "$existing_ip" "$new_ip" "$existing_name" "$new_name" "$tmpf"
            cat "$tmpf" >> "$COLLECTED"
            log "  -> $(head -1 "$tmpf" | cut -d, -f3) Mbps, $(head -1 "$tmpf" | cut -d, -f4) ms"
        done
    done

    # Also measure between new nodes themselves (if >1)
    for (( a=1; a<=${#NEW_NODES[@]}; a++ )); do
        for (( b=a+1; b<=${#NEW_NODES[@]}; b++ )); do
            na="${NEW_NODES[$a]}"; nb="${NEW_NODES[$b]}"
            ia=$(ip_of "$na"); ib=$(ip_of "$nb")
            log "Measuring $na <-> $nb..."
            tmpf="$TMPDIR_MEAS/${na}_${nb}.csv"
            measure_pair "$ia" "$ib" "$na" "$nb" "$tmpf"
            cat "$tmpf" >> "$COLLECTED"
            log "  -> $(head -1 "$tmpf" | cut -d, -f3) Mbps, $(head -1 "$tmpf" | cut -d, -f4) ms"
        done
    done

    merge_into_csv "$COLLECTED" "$CSV"
    log "Done. Merged results into $CSV"
    exit 0
fi

# ===========================================================================
# MODE: full — measure everything, write fresh CSV
# ===========================================================================

# Check iperf3 on all nodes
log "Checking iperf3 availability..."
for rpi in "${RPIS[@]}"; do
    if ! ssh_cmd "$rpi" "which iperf3 > /dev/null 2>&1"; then
        echo "ERROR: iperf3 not found on $rpi. Run: sudo apt install -y iperf3" >&2; exit 1
    fi
done

mkdir -p "$(dirname "$CSV")"
echo "device1,device2,bw_mbps,rtt_ms" > "$CSV"
total_pairs=$(( N + N * (N - 1) / 2 ))
print_estimate $total_pairs 0
log "Writing results to $CSV"
$PARALLEL && log "Mode: PARALLEL" || log "Mode: serial"

N=${#RPIS[@]}
TMPDIR_MEAS=$(mktemp -d)
trap 'rm -rf "$TMPDIR_MEAS"' EXIT

if ! $PARALLEL; then
    for (( k=1; k<=N; k++ )); do
        rpi="${RPIS[$k]}"; rpi_name="${RPI_NAMES[$k]}"
        log "Measuring client <-> $rpi_name ($rpi)..."
        tmpf="$TMPDIR_MEAS/c_${k}.csv"
        measure_pair "client" "$rpi" "client" "$rpi_name" "$tmpf"
        cat "$tmpf" >> "$CSV"
        log "  -> $(head -1 "$tmpf" | cut -d, -f3) Mbps, $(head -1 "$tmpf" | cut -d, -f4) ms"
    done

    for (( i=1; i<=N; i++ )); do
        for (( j=i+1; j<=N; j++ )); do
            log "Measuring ${RPI_NAMES[$i]} <-> ${RPI_NAMES[$j]}..."
            tmpf="$TMPDIR_MEAS/p_${i}_${j}.csv"
            measure_pair "${RPIS[$i]}" "${RPIS[$j]}" "${RPI_NAMES[$i]}" "${RPI_NAMES[$j]}" "$tmpf"
            cat "$tmpf" >> "$CSV"
            log "  -> $(head -1 "$tmpf" | cut -d, -f3) Mbps, $(head -1 "$tmpf" | cut -d, -f4) ms"
        done
    done

else
    # Client -> all RPis in parallel
    log "Measuring client <-> all RPis in parallel..."
    for (( k=1; k<=N; k++ )); do
        tmpf="$TMPDIR_MEAS/c_${k}.csv"
        measure_pair "client" "${RPIS[$k]}" "client" "${RPI_NAMES[$k]}" "$tmpf" &
    done
    wait || true
    for (( k=1; k<=N; k++ )); do
        tmpf="$TMPDIR_MEAS/c_${k}.csv"
        [[ -f "$tmpf" ]] && cat "$tmpf" >> "$CSV"
    done

    # Build pair list and batch greedily (no associative arrays — use a string set)
    remaining=()
    for (( i=1; i<=N; i++ )); do
        for (( j=i+1; j<=N; j++ )); do
            remaining+=("$i $j")
        done
    done

    batch_num=0
    while [[ ${#remaining[@]} -gt 0 ]]; do
        (( batch_num++ )) || true
        batch_pairs=()
        next_remaining=()
        busy_str=""   # space-separated list of busy indices

        for pair in "${remaining[@]}"; do
            i="${pair% *}"; j="${pair#* }"
            if [[ " $busy_str " != *" $i "* && " $busy_str " != *" $j "* ]]; then
                batch_pairs+=("$pair")
                busy_str="$busy_str $i $j"
            else
                next_remaining+=("$pair")
            fi
        done

        desc=""
        for pair in "${batch_pairs[@]}"; do
            i="${pair% *}"; j="${pair#* }"
            desc+="${RPI_NAMES[$i]}<->${RPI_NAMES[$j]}  "
        done
        log "Parallel batch $batch_num (${#batch_pairs[@]} pairs): $desc"

        for pair in "${batch_pairs[@]}"; do
            i="${pair% *}"; j="${pair#* }"
            tmpf="$TMPDIR_MEAS/p_${i}_${j}.csv"
            measure_pair "${RPIS[$i]}" "${RPIS[$j]}" "${RPI_NAMES[$i]}" "${RPI_NAMES[$j]}" "$tmpf" &
        done
        wait || true

        for pair in "${batch_pairs[@]}"; do
            i="${pair% *}"; j="${pair#* }"
            tmpf="$TMPDIR_MEAS/p_${i}_${j}.csv"
            [[ -f "$tmpf" ]] && cat "$tmpf" >> "$CSV"
        done

        remaining=("${next_remaining[@]}")
    done
fi

log "Done. Results written to $CSV"