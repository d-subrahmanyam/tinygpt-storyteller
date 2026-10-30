#!/usr/bin/env python3
"""Stress test for the RX 7900 XT eGPU over M.2.

Loops three checks until the time is up and counts any data errors:
  - compute: large FP32 matmuls, checked against a reference result
  - VRAM:    fills a big chunk of VRAM with a pattern and verifies it
  - PCIe:    copies pinned buffers host->GPU->host and compares them
A sensor line (power, temps, load, link) is printed every interval.

Usage: uv run tools/egpu_stress.py [seconds] (default 600)
"""
import glob
import os
import sys
import threading
import time

import torch

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 600
REPORT_EVERY = 15
MATMUL_N = 8192
PCIE_BYTES = 512 * 1024**2
CHUNK = 64 * 1024**2  # int32 elements per VRAM verify chunk (256 MiB)


def find_sysfs():
    for dev in glob.glob("/sys/bus/pci/devices/*"):
        try:
            v = open(f"{dev}/vendor").read().strip()
            d = open(f"{dev}/device").read().strip()
        except OSError:
            continue
        if v == "0x1002" and d == "0x744c":
            return dev
    return None


def read(path, default="?"):
    try:
        return open(path).read().strip()
    except OSError:
        return default


def sensors(dev):
    hw = (glob.glob(f"{dev}/hwmon/hwmon*") or [""])[0]
    watts = int(read(f"{hw}/power1_average", "0")) / 1e6
    temps = {}
    for i in (1, 2, 3):
        label = read(f"{hw}/temp{i}_label", f"t{i}")
        temps[label] = int(read(f"{hw}/temp{i}_input", "0")) // 1000
    busy = read(f"{dev}/gpu_busy_percent")
    # The card sits behind its own PCIe switch; walk up to the root port's link.
    root = os.path.realpath(dev)
    while os.path.basename(os.path.dirname(root)).startswith("0000:"):
        root = os.path.dirname(root)
    link = f"{read(f'{root}/current_link_speed')} x{read(f'{root}/current_link_width')}"
    return watts, temps, busy, link


def pick_device():
    for i in range(torch.cuda.device_count()):
        if "7900" in torch.cuda.get_device_name(i):
            return i
    sys.exit("RX 7900 XT not found by PyTorch")


def main():
    dev_sysfs = find_sysfs()
    idx = pick_device()
    torch.cuda.set_device(idx)
    dev = torch.device("cuda", idx)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}, "
          f"{torch.cuda.get_device_properties(idx).total_memory / 2**30:.1f} GiB, "
          f"running {DURATION}s", flush=True)

    g = torch.Generator(device=dev).manual_seed(1234)
    a = torch.randn(MATMUL_N, MATMUL_N, device=dev, generator=g)
    b = torch.randn(MATMUL_N, MATMUL_N, device=dev, generator=g)
    ref = a @ b
    torch.cuda.synchronize()

    free, _ = torch.cuda.mem_get_info()
    vram_elems = int((free - 3 * 2**30) // 4 // CHUNK * CHUNK)
    vram = torch.empty(vram_elems, dtype=torch.int32, device=dev)
    print(f"VRAM test buffer: {vram_elems * 4 / 2**30:.1f} GiB", flush=True)

    h_src = torch.randint(0, 256, (PCIE_BYTES,), dtype=torch.uint8).pin_memory()
    h_dst = torch.empty_like(h_src).pin_memory()
    d_buf = torch.empty(PCIE_BYTES, dtype=torch.uint8, device=dev)

    errors = {"compute": 0, "vram": 0, "pcie": 0}
    stats = {"tflops": [], "h2d": [], "d2h": [], "cycles": 0}
    stop = threading.Event()
    peak = {"watts": 0.0, "junction": 0}

    def monitor():
        last = 0.0
        start = time.time()
        while not stop.is_set():
            w, t, busy, link = sensors(dev_sysfs)
            peak["watts"] = max(peak["watts"], w)
            peak["junction"] = max(peak["junction"], t.get("junction", 0))
            if time.time() - last >= REPORT_EVERY:
                last = time.time()
                tf = stats["tflops"][-1] if stats["tflops"] else 0
                h2d = stats["h2d"][-1] if stats["h2d"] else 0
                d2h = stats["d2h"][-1] if stats["d2h"] else 0
                temps = " ".join(f"{k}={v}C" for k, v in t.items())
                print(f"[{int(time.time() - start):4d}s] {w:5.0f}W {temps} busy={busy}% "
                      f"link={link} | {tf:5.1f} TFLOPS h2d={h2d:4.1f} d2h={d2h:4.1f} GB/s "
                      f"| errors={sum(errors.values())}", flush=True)
            time.sleep(1)

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    t_end = time.time() + DURATION
    seed = 0
    try:
        while time.time() < t_end:
            seed += 1
            # Compute
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = 10
            for _ in range(reps):
                c = a @ b
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            stats["tflops"].append(2 * MATMUL_N**3 * reps / dt / 1e12)
            if not torch.equal(c, ref):
                diff = (c - ref).abs().max().item()
                if diff > 1e-2:
                    errors["compute"] += 1
                    print(f"!! compute mismatch, max diff {diff}", flush=True)

            # VRAM: write a seed-dependent pattern, then verify it chunk by chunk
            # Pattern uses the in-chunk index so int32 never has to hold the full offset.
            mult = 2654435761 & 0x7FFFFFFF
            base = torch.arange(CHUNK, dtype=torch.int32, device=dev) * mult
            for k, s in enumerate(range(0, vram_elems, CHUNK)):
                vram[s:s + CHUNK] = base + (k * 7919 + seed)
            for k, s in enumerate(range(0, vram_elems, CHUNK)):
                bad = (vram[s:s + CHUNK] != base + (k * 7919 + seed)).sum().item()
                if bad:
                    errors["vram"] += bad
                    print(f"!! {bad} VRAM mismatches in chunk at {s}", flush=True)

            # PCIe round trips
            for _ in range(4):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                d_buf.copy_(h_src, non_blocking=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                h_dst.copy_(d_buf, non_blocking=True)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                stats["h2d"].append(PCIE_BYTES / (t1 - t0) / 1e9)
                stats["d2h"].append(PCIE_BYTES / (t2 - t1) / 1e9)
                if not torch.equal(h_src, h_dst):
                    errors["pcie"] += 1
                    print("!! PCIe round-trip data mismatch", flush=True)
                h_dst.zero_()
            stats["cycles"] += 1
    except RuntimeError as e:
        stop.set()
        print(f"!! GPU runtime error: {e}", flush=True)
        sys.exit(2)
    stop.set()
    mon.join(timeout=3)

    avg = lambda xs: sum(xs) / len(xs) if xs else 0
    print("\n=== Summary ===")
    print(f"cycles:        {stats['cycles']}")
    print(f"compute:       {avg(stats['tflops']):.1f} TFLOPS FP32 avg")
    print(f"PCIe h2d/d2h:  {avg(stats['h2d']):.2f} / {avg(stats['d2h']):.2f} GB/s avg")
    print(f"peak power:    {peak['watts']:.0f} W, peak junction {peak['junction']} C")
    print(f"errors:        {errors}")
    print("RESULT:", "PASS" if not any(errors.values()) else "FAIL")
    sys.exit(0 if not any(errors.values()) else 1)


if __name__ == "__main__":
    main()
