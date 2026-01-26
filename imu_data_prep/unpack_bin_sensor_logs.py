# unpack_bin_sensor_logs.py
"""
# Unpacks Bin Sensor Logs created by log_linacc_quat_gyro_flash_spi.py

Input data:
    *** CAUTION: TIME IN msec NOT SECONDS, for BNO086 efficiency at 5ms 200Hz
    it reads the binary data in "flight_log_2026xxxx_xpm_sector.bin" which has the
    fp32 data following order:
        ts_ms, ax, ay, az, qr, qi, qj, qk, gy, gp, gr

    Wnich was collected with:
        ax, ay, az, acc, ts_ms = bno.linear_acceleration.full
        qr, qi, qj, qk = bno.quaternion
        gy, gp, gr = bno.gyro

Output:
    *** CAUTION: The created CSV IS CONVERTED TO SECONDS by unpack_bin_sensor_logs.py
    SECONDS, ax, ay, az, qr, qi, qj, qk, gy, gp, gr


The format of the file for  both ways of reading/storing results are identical:
    1. simple whole file where rows are contiguous
    2. 4 KiB sector-sized chunks with 96 rows of data and 4 bytes CRC
       This code checks the CRC and will error and skip bad sectors.

Sample timings:
# Reading Linear_Acc for 1000 rows, doing nothing with output
# sensor timestamps last_sensor_ms=5724.8 first_sensor_ms=644.9  sensor duration: 5.1 s
# Sensor msec/Lin_Acc = 5.08 ms
# Clock msec/Lin_Acc  = 5.06 ms

Sector-formatted data has delays caused by sector writes & flushes

Decoded 1000 rows, FILE OK, No Corrupt blocks: 0)
	Header: ['ts_ms', 'ax', 'ay', 'az', 'qr', 'qi', 'qj', 'qk', 'gy', 'gp', 'gr']
	Data: [6274.5000, 0.0195, -0.0156, 0.0117, 0.9928, 0.0104, -0.0200, 0.1179, 0.0000, 0.0000, 0.0000]
	Data: [6295.7002, 0.0195, -0.0156, 0.0117, 0.9928, 0.0104, -0.0200, 0.1179, 0.0000, 0.0000, 0.0000]
	Data: [6301.0000, -0.0273, 0.0000, -0.0078, 0.9928, 0.0104, -0.0200, 0.1179, 0.0000, 0.0000, 0.0000]
	Data: [6306.7998, 0.0000, -0.0117, 0.0195, 0.9928, 0.0104, -0.0200, 0.1179, 0.0000, 0.0000, 0.0000]
	Data: [6311.0000, 0.0312, 0.0039, 0.0273, 0.9928, 0.0104, -0.0200, 0.1179, 0.0000, 0.0000, 0.0000]

Average Data Freq: 164.63 Hz
	Average time step: 6.07 ms
	Min/Max interval: 1.50 / 105.50 ms
	Std Dev: 8.62 ms, jitter: 141.9%

Histogram of report periods (ms):
 0.0– 1.0 (    0) |
 1.0– 2.0 (    4) | #
 2.0– 3.0 (   10) | #
 3.0– 4.0 (   41) | ####
 4.0– 5.0 (  385) | ##########################################
 5.0– 6.0 (  457) | ##################################################
 6.0– 7.0 (   58) | ######
 7.0– 8.0 (    6) | #
 8.0– 9.0 (    7) | #
 9.0–10.0 (    2) | #
10.0–11.0 (    4) | #
11.0–12.0 (    2) | #
12.0–13.0 (    0) |
13.0–14.0 (    0) |
14.0–15.0 (    0) |
15.0–16.0 (    0) |
16.0–17.0 (    1) | #
17.0–18.0 (    0) |
18.0–19.0 (    0) |
19.0–20.0 (    0) |
    >20.0 (   22) | ##
"""
import binascii
import os
import struct

import numpy as np

# Constants matching Pico code: log_linacc_quat_gyro_flash.py
NUM_FLOATS = 11
BYTES_PER_ROW = 44
ROWS_PER_SECTOR = 93
DATA_PART_SIZE = 4092  # 93 rows * 44 bytes
SECTOR_SIZE = 4096  # 4KiB

ROW_DTYPE = np.dtype("<f4", NUM_FLOATS)


def decode_whole_buffer(filename):
    print(f"\n--- Decoding Whole Buffer File: {filename}")

    with open(filename, "rb") as f:
        raw = f.read()

    # Drop incomplete trailing bytes
    nrows = len(raw) // BYTES_PER_ROW
    raw = raw[: nrows * BYTES_PER_ROW]

    data = np.frombuffer(raw, dtype=ROW_DTYPE).reshape(-1, NUM_FLOATS)

    print(f"Decoded {data.shape[0]} rows")
    return data


def decode_sector_buffer(filename):
    print(f"\n--- Decoding Sector Buffer File (CRC Verified): {filename}")

    rows = []
    corrupt_blocks = 0

    with open(filename, "rb") as f:
        block_idx = 0
        while True:
            sector = f.read(SECTOR_SIZE)
            if not sector:
                break
            if len(sector) < SECTOR_SIZE:
                print(f"Warning: Final sector is incomplete ({len(sector)} bytes). Skipping sector.")
                break

            data_part = sector[:DATA_PART_SIZE]
            stored_crc = struct.unpack("<I", sector[DATA_PART_SIZE:SECTOR_SIZE])[0]
            computed_crc = binascii.crc32(data_part) & 0xFFFFFFFF

            if stored_crc != computed_crc:
                print(f"Warning: CRC FAIL! at Block {block_idx}: ")
                corrupt_blocks += 1
                block_idx += 1
                continue

            block = np.frombuffer(data_part, dtype=ROW_DTYPE).reshape(-1, NUM_FLOATS)

            # Drop all zero rows signal of mid-sector termination, ts_ms will never be 0
            valid = np.any(block != 0.0, axis=1)
            block = block[valid]

            rows.append(block)
            block_idx += 1

    if rows:
        data = np.vstack(rows)
    else:
        data = np.empty((0, NUM_FLOATS), dtype=np.float32)

    if corrupt_blocks == 0:
        print(f"Decoded {data.shape[0]} rows, FILE OK, No Corrupt blocks: {corrupt_blocks})")
    else:
        print(f"Decoded {data.shape[0]} rows, {corrupt_blocks} Corrupt blocks")

    return data


def ascii_histogram(data):
    """
    Prints horizontal ascii histogram showing times andcounts. Horizontal scale of longest bar is max_width.
    :param data:
    :return:
    """

    # 5ms samples
    # define bins: 0.0–1.0, 1.0–2.0, ..., 9.0–10.0, >10.0
    bin_min = 0.0
    bin_max = 21.0

    # 100ms samples
    bin_min = 0.0
    bin_max = 200.0
    bin_width = 20.0

    bins = np.arange(bin_min,  bin_max+bin_width, bin_width)
    counts, _ = np.histogram(data, bins=bins)

    # overflow (>10.0)
    overflow = np.sum(data > bin_max)

    # limit histogram lenth
    max_width = 50
    all_counts = np.append(counts, overflow)
    max_count = all_counts.max()
    scale = max_width / max_count if max_count > 0 else 1.0

    def bar(n):
        """ string of bar lengths, at least 1 for non-zero counts """
        if n == 0:
            return ""
        return "#" * max(1, int(round(n * scale)))

    # print histogram
    for i, count in enumerate(counts):
        left = bins[i]
        right = bins[i + 1]
        label = f"{left:5.1f}–{right:5.1f} ({count:5d})"
        print(f"{label} | {bar(count)}")

    print(f"     >{bin_max} ({overflow:5d}) | {bar(overflow)}")


def print_summary(data):
    """ print sample of first rows, and print timestamp statistics with ascii histogram"""
    if data.size == 0:
        print("No data")
        return

    h = ["ts_ms", "ax", "ay", "az", "qr", "qi", "qj", "qk", "gy", "gp", "gr"]
    print(f"\tHeader: {h}")

    for i in range(min(5, data.shape[0])):
        row = ", ".join(f"{x:.4f}" for x in data[i])
        print(f"\tData: [{row}]")

    time_ms = data[:, 0]
    deltas = np.diff(time_ms)

    dt_avg = deltas.mean()
    dt_min = deltas.min()
    dt_max = deltas.max()
    dt_std = deltas.std()
    sample_frequency = 1.0 / (dt_avg / 1000.0)

    print(f"\nAverage Data Freq: {sample_frequency:.2f} Hz")
    print(f"\tAverage time step: {dt_avg :.2f} ms")
    print(f"\tMin/Max interval: {dt_min:.2f} / {dt_max:.2f} ms")
    print(f"\tStd Dev: {dt_std:.2f} ms, jitter: {(dt_std / dt_avg) * 100:.1f}%")

    print("\nHistogram of report periods (ms):")
    ascii_histogram(deltas)


def write_csv(filename, data, precision=7):
    """ Write sensor data to CSV"""
    if data.size == 0:
        print(f"No data to write: {filename}")
        return

    header = ["SECONDS", "ax", "ay", "az", "qr", "qi", "qj", "qk", "gy", "gp", "gr", ]
    fmt = ["%.4f"] + [f"%.{precision}f"] * (data.shape[1] - 1)

    print("\n*** WARNING: CSV is converted to SECONDS, above processing in msec")
    data = np.array(data, copy=True)
    time_ms = data[:, 0].copy()
    data[:, 0] = time_ms / 1000.0

    np.savetxt(filename, data, delimiter=",", header=",".join(header), comments="", fmt=fmt)

    print(f"\nWrote CSV: {filename} ({data.shape[0]} rows)")


# Main #############################################
def main():
    # Convert Whole-Data-formatted file
    filename = "data_logs/flight_log_2026xxxx_xpm_whole.bin"
    if os.path.exists(filename):
        whole_data = decode_whole_buffer(filename)
        print_summary(whole_data)
        write_csv(filename.replace(".bin", ".csv"), whole_data)
    else:
        print(f"\nFile {filename} does not exist")

    # Convert Sector-formatted file
    filename = "data_logs/flight_log_2026xxxx_xpm_sector.bin"
    if os.path.exists(filename):
        sector_data = decode_sector_buffer(filename)
        print_summary(sector_data)
        write_csv(filename.replace(".bin", ".csv"), sector_data)
    else:
        print(f"\nError: File {filename} does not exist")


if __name__ == "__main__":
    main()
