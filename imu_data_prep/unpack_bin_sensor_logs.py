# unpack_bin_sensor_logs.py
"""
# Unpacks Bin Sensor Logs created by log_linacc_quat_gyro_flash_spi.py
#
# Unpacks both styles of writes:
# 1. simple whole file where rows are contiguous
# 2. 4 KiB sector-sized chunks with 96 rows of data and 4 bytes CRC
#
# Sample timings:
# Reading Linear_Acc for 1000 rows, doing nothing with output
# sensor timestamps last_sensor_ms=5724.8 first_sensor_ms=644.9  sensor duration: 5.1 s
# Sensor msec/Lin_Acc = 5.08 ms
# Clock msec/Lin_Acc  = 5.06 ms
#
# Writing data in sector chunks to flash in 4096 sectors with flush no CRC
# Array each result for 1000 rows:
# sensor timestamps last_sensor_ms=11931.5 first_sensor_ms=5724.8  sensor duration: 6.2 s
# Sensor msec/reports = 6.21 ms
# Clock msec/reports  = 6.28 ms
# BYTES_PER_ROW=44, data size = 44000 bytes
# Array = 43.0 KiB, xfer = 6.8 KiB/s
#
# Writing data in sector chunks to flash in 4096 sectors with flush WITH CRC
# Array each result for 1000 rows:
# sensor timestamps last_sensor_ms=11997.0 first_sensor_ms=5785.0  sensor duration: 6.2 s
# Sensor msec/reports = 6.21 ms
# Clock msec/reports  = 6.29 ms
# BYTES_PER_ROW=44, data size = 44000 bytes
# Array = 43.0 KiB, xfer = 6.8 KiB/s

--- Decoding Whole Buffer File: sensor_log.bin
Decoded 1000 rows
Header: ['ts_ms', 'ax', 'ay', 'az', 'qr', 'qi', 'qj', 'qk', 'gy', 'gp', 'gr']
Data: [np.float32(657.6), np.float32(-0.0039), np.float32(0.0234), np.float32(-0.0078), np.float32(1.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(-0.0117), np.float32(0.0059)]
Data: [np.float32(679.1), np.float32(0.1562), np.float32(0.1289), np.float32(0.1719), np.float32(1.0), np.float32(0.0), np.float32(-1e-04), np.float32(0.0), np.float32(0.0059), np.float32(-0.002), np.float32(0.0039)]
Data: [np.float32(683.0), np.float32(0.125), np.float32(0.1328), np.float32(0.1758), np.float32(0.9996), np.float32(0.0217), np.float32(-0.0167), np.float32(-0.0004), np.float32(0.0059), np.float32(-0.0098), np.float32(0.0)]
Data: [np.float32(688.4), np.float32(0.1406), np.float32(0.1445), np.float32(0.1836), np.float32(0.9996), np.float32(0.0217), np.float32(-0.0168), np.float32(-0.0004), np.float32(-0.002), np.float32(-0.0176), np.float32(0.0039)]
Data: [np.float32(693.0), np.float32(0.1836), np.float32(0.1641), np.float32(0.1836), np.float32(0.9996), np.float32(0.0218), np.float32(-0.0169), np.float32(-0.0004), np.float32(0.0), np.float32(-0.0195), np.float32(0.0)]
	Average Data Freq: 197.59 Hz
	Average time step: 5.06 ms
	Min/Max interval: 1.70 / 28.00 ms
	Std Dev: 1.30 ms, jitter: 25.7%

--- Decoding Sector Buffer File (CRC Verified): flight_log_2026xxxx_xpm_sector.bin
Decoded 1000 rows. (Corrupt blocks: 0)
Header: ['ts_ms', 'ax', 'ay', 'az', 'qr', 'qi', 'qj', 'qk', 'gy', 'gp', 'gr']
Data: [np.float32(6248.0), np.float32(0.0039), np.float32(0.0117), np.float32(0.0195), np.float32(0.9992), np.float32(0.0294), np.float32(-0.027), np.float32(-0.0002), np.float32(0.0), np.float32(0.0), np.float32(0.0)]
Data: [np.float32(6265.2), np.float32(0.0039), np.float32(0.0117), np.float32(0.0195), np.float32(0.9992), np.float32(0.0294), np.float32(-0.027), np.float32(-0.0002), np.float32(0.0), np.float32(0.0), np.float32(0.0)]
Data: [np.float32(6269.4), np.float32(0.0), np.float32(-0.0312), np.float32(0.0), np.float32(0.9992), np.float32(0.0294), np.float32(-0.027), np.float32(-0.0002), np.float32(0.0), np.float32(0.0), np.float32(0.0)]
Data: [np.float32(6275.0), np.float32(0.043), np.float32(0.0234), np.float32(0.0078), np.float32(0.9992), np.float32(0.0294), np.float32(-0.027), np.float32(-0.0002), np.float32(0.0), np.float32(0.0), np.float32(0.0)]
Data: [np.float32(6280.0), np.float32(-0.0039), np.float32(0.0039), np.float32(-0.0117), np.float32(0.9992), np.float32(0.0294), np.float32(-0.027), np.float32(-0.0002), np.float32(0.0), np.float32(0.0), np.float32(0.0)]
	Average Data Freq: 167.68 Hz
	Average time step: 5.96 ms
	Min/Max interval: 2.90 / 108.70 ms
	Std Dev: 8.08 ms, jitter: 135.5%
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


def print_summary(data):
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


def write_csv(filename, data, precision=7):
    """ Write sensor data to CSV"""
    if data.size == 0:
        print(f"No data to write: {filename}")
        return

    header = ["ts_ms", "ax", "ay", "az", "qr", "qi", "qj", "qk", "gy", "gp", "gr", ]
    fmt = ["%.1f"] + [f"%.{precision}f"] * (data.shape[1] - 1)

    np.savetxt(filename, data, delimiter=",", header=",".join(header), comments="", fmt=fmt)

    print(f"\nWrote CSV: {filename} ({data.shape[0]} rows)")


# Main #############################################
def main():

    # Convert Whole Data formatted file
    filename = "data_logs/flight_log_2026xxxx_xpm_whole.bin"
    if os.path.exists(filename):
        raw_data = decode_whole_buffer(filename)
        print_summary(raw_data)

    # Conver Sector-formatted file
    filename = "data_logs/flight_log_2026xxxx_xpm_sector.bin"
    if os.path.exists(filename):
        sector_data = decode_sector_buffer(filename)
        print_summary(sector_data)
        write_csv(filename.replace(".bin", ".csv"), sector_data)


if __name__ == "__main__":
    main()
