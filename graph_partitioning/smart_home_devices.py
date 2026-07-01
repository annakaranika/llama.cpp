"""Realistic smart-home device pools for the simulator.

Three named clusters scaled to the sizes used throughout the project:

* SMALL  (4 devices)  — minimal smart home: TV + speaker + router + thermostat.
* MEDIUM (32 devices) — enthusiast: multiple TVs / speakers / cameras, hub,
                        appliances, a couple of tablets and a laptop.
* LARGE  (48 devices) — heavy automation: adds per-room sensors, more
                        cameras, a NAS, multiple mesh nodes, etc.

Compute (GFLOPS) values are *effective sustained throughput for transformer
matmul*, not peak NPU/GPU numbers. The simulator uses these as a single
scalar, so we collapse "this device can NPU INT8 nicely" into a higher
GFLOPS value. Memory is real DRAM that the OS would let us pin for
inference. Swap bandwidth corresponds to the local storage tier (SD card,
eMMC, NVMe).

Source of intuition (rough):
  * RPi 4 (CPU): 3-5 GFLOPS effective FP32 matmul
  * RPi 5 (CPU): ~10 GFLOPS
  * Smart-TV SoCs with NPUs (LG/Samsung mid): 5-20 GFLOPS effective
  * Apple TV 4K / Nvidia Shield: 30-50 GFLOPS effective (NPU/GPU)
  * Smart speakers (Echo / Nest): 2-6 GFLOPS
  * Wi-Fi routers (Cortex-A53/A57): 1-5 GFLOPS
  * Smart fridges (Samsung Family Hub): ~5 GFLOPS, 1-2 GB
  * Tablets / phones idling: 50-200 GFLOPS via GPU/NPU
  * Laptops at idle: 100-300 GFLOPS sustained

Storage tiers:
  * SD card: ~50 Mbps
  * eMMC: ~200-400 Mbps
  * NVMe SSD (tablet / laptop): 1500-4000 Mbps
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .common import Device


# Each entry: (display_name, gflops, memory_gb, swap_mbps)
_DeviceSpec = Tuple[str, float, float, float]


SMALL_HOME: List[_DeviceSpec] = [
    # Mid-range smart TV with NPU (e.g. LG OLED, Samsung QLED) — 4 GB RAM
    ("smart_tv_living",  10.0, 4.0, 300),
    # Standard smart speaker (Echo / Nest Audio / HomePod mini) — 1 GB RAM
    ("smart_speaker",     4.0, 1.0, 200),
    # Mesh Wi-Fi 6 router (e.g. Eero Pro 6E) — 1 GB RAM
    ("router_wifi6",      4.0, 1.0, 300),
    # Smart thermostat (Nest Learning) — 512 MB
    ("thermostat",        0.5, 0.5,  50),
]


MEDIUM_HOME: List[_DeviceSpec] = [
    # Displays / media
    ("smart_tv_living",     10.0, 4.0, 300),   # mid-range OLED
    ("smart_tv_bedroom",     8.0, 3.0, 200),   # standard 4K smart TV
    ("smart_tv_kitchen",     5.0, 2.0, 200),   # small kitchen display TV
    ("apple_tv_4k",         40.0, 4.0, 500),   # Apple TV 4K (3rd gen, A15)
    ("smart_display_kit",    8.0, 2.0, 250),   # Nest Hub Max / Echo Show 10
    # Speakers (most real smart speakers have 1-2 GB now)
    ("speaker_kitchen",      4.0, 1.0, 200),   # Echo / Nest Audio
    ("speaker_bedroom",      4.0, 1.0, 200),
    ("speaker_bathroom",     2.0, 0.5, 100),   # Echo Dot mini / Nest Mini
    ("speaker_garage",       2.0, 0.5, 100),
    # Network
    ("router_main",          4.0, 1.0, 300),   # Eero Pro / ASUS AX6000
    ("mesh_node_upstairs",   3.0, 1.0, 200),
    ("mesh_node_garage",     3.0, 1.0, 200),
    # Hub
    ("home_hub_pi5",         8.0, 4.0, 400),   # Home Assistant on RPi 5
    # Cameras / security
    ("cam_front_door",       2.0, 1.0, 100),   # Nest Doorbell (battery): 1 GB
    ("cam_back_yard",        2.0, 1.0, 100),
    ("cam_garage",           1.5, 0.5, 100),   # Cheaper Ring cam: 512 MB
    ("video_doorbell",       1.5, 0.5, 100),   # Ring Doorbell Pro: 512 MB
    # HVAC / sensors with screens
    ("thermostat_main",      0.5, 0.5,  50),
    ("thermostat_upstairs",  0.5, 0.5,  50),
    # Kitchen appliances
    ("smart_fridge",         5.0, 4.0, 200),   # Samsung Family Hub
    ("smart_oven",           1.0, 0.5, 100),
    ("smart_dishwasher",     0.5, 0.25, 50),   # tiny MCU class
    # Laundry
    ("smart_washer",         0.5, 0.5,  50),
    ("smart_dryer",          0.5, 0.5,  50),
    # Misc smart appliances (mostly MCUs, won't host model shards)
    ("smart_lock",           0.2, 0.25, 50),
    ("smart_garage_opener",  0.8, 0.5, 100),
    ("smart_blinds",         0.3, 0.25, 50),
    # Idling personal devices (opportunistically available)
    ("tablet_kitchen",     150.0, 4.0, 1500),
    ("tablet_bedroom",     120.0, 4.0, 1200),
    ("phone_dock_old",      80.0, 3.0, 1000),
    # The "compute peak" of the house
    ("laptop_idle",        200.0, 8.0, 3000),
    ("desktop_idle",       300.0, 16.0, 4000),
]


LARGE_HOME: List[_DeviceSpec] = MEDIUM_HOME + [
    # Extra room displays
    ("smart_tv_basement",    6.0, 2.0, 200),
    ("smart_tv_guest",       5.0, 2.0, 200),
    ("smart_tv_office",      8.0, 3.0, 300),
    # More cameras + sensors
    ("cam_living_room",      2.0, 1.0, 100),
    ("cam_basement",         1.5, 0.5, 100),
    ("cam_baby_monitor",     3.0, 1.0, 200),
    ("cam_dog_monitor",      2.0, 1.0, 100),
    # Multi-zone HVAC
    ("thermostat_basement",  0.5, 0.5,  50),
    ("thermostat_garage",    0.5, 0.5,  50),
    # Outdoor / irrigation (MCU class)
    ("irrigation_ctrl",      0.4, 0.25, 50),
    ("pool_pump_ctrl",       0.3, 0.25, 50),
    # Power monitoring (low-end Cortex-M)
    ("home_battery_inv",     0.5, 0.5, 100),
    ("solar_inverter",       0.5, 0.5, 100),
    # Storage tier — the NAS in the house, usable for inference too
    ("nas_box",             50.0, 16.0, 4000),  # Synology DS923+ / similar
    # Extra devices for a heavy household
    ("phone_office",        80.0, 3.0, 1000),
    ("tablet_office",      120.0, 4.0, 1200),
]


_PRESETS: Dict[str, List[_DeviceSpec]] = {
    "small":  SMALL_HOME,    # 4 devices
    "medium": MEDIUM_HOME,   # 32 devices
    "large":  LARGE_HOME,    # 48 devices
}


def list_preset_names() -> List[str]:
    return list(_PRESETS.keys())


def smart_home_servers(preset: str) -> List[Device]:
    """Return the Device list for the named smart-home preset.

    Args:
        preset: one of "small" (4 devices), "medium" (32), "large" (48).
    """
    if preset not in _PRESETS:
        raise KeyError(
            f"Unknown smart-home preset {preset!r}; choices are "
            f"{list(_PRESETS.keys())}"
        )
    specs = _PRESETS[preset]
    return [
        Device(
            name=name,
            gflops=gflops,
            memory_gb=mem_gb,
            swap_bandwidth_mbps=swap_mbps,
        )
        for (name, gflops, mem_gb, swap_mbps) in specs
    ]


def preset_summary(preset: str) -> Dict[str, float]:
    """Aggregate stats for a preset (helpful for printing context)."""
    devs = smart_home_servers(preset)
    return {
        "n_devices": len(devs),
        "total_gflops": sum(d.gflops for d in devs),
        "total_dram_gb": sum(d.memory_gb for d in devs),
        "min_gflops": min(d.gflops for d in devs),
        "max_gflops": max(d.gflops for d in devs),
        "min_dram_gb": min(d.memory_gb for d in devs),
        "max_dram_gb": max(d.memory_gb for d in devs),
        "median_gflops": sorted(d.gflops for d in devs)[len(devs) // 2],
        "median_dram_gb": sorted(d.memory_gb for d in devs)[len(devs) // 2],
    }
