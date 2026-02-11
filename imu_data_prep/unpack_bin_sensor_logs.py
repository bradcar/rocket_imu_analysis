# unpack_bin_sensor_logs.py
"""
# Unpacks Bin Sensor Logs created by log_linacc_quat_gyro_flash_spi.py

Input data:
    *** CAUTION: TIME IN msec NOT SECONDS, for BNO086 efficiency at 5ms 200Hz
    it reads the binary data in "flight_log_2026xxxx_xpm_sector.bin" which has the
    fp32 data following order:
        ts_ms, ax, ay, az, qr, qi, qj, qk, gy, gp, gr, hpa

    Wnich was collected with:
        ax, ay, az, acc, ts_ms = bno.linear_acceleration.full
        qr, qi, qj, qk = bno.quaternion
        gy, gp, gr = bno.gyro
        hpa = bmp.pressure

Output:
    *** CAUTION: The created CSV IS CONVERTED TO SECONDS by unpack_bin_sensor_logs.py
    SECONDS, ax, ay, az, qr, qi, qj, qk, gy, gp, gr, hpa


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
	Header: ['ts_ms', 'ax', 'ay', 'az', 'qr', 'qi', 'qj', 'qk', 'gy', 'gp', 'gr', 'hpa']
	Data: [17170.0000, 0.0195, 0.0391, -0.0195, 0.6223, -0.0134, -0.0254, -0.7822, 0.0000, 0.0000, 0.0000, 99.9000]
	Data: [17185.1992, 0.0195, 0.0391, -0.0195, 0.6223, -0.0134, -0.0254, -0.7822, 0.0000, 0.0000, 0.0000, 99.9000]
	Data: [17190.0000, -0.0273, 0.0000, 0.0039, 0.6223, -0.0134, -0.0253, -0.7822, -0.0039, 0.0098, 0.0000, 99.9000]
	Data: [17195.1992, -0.0156, -0.0117, 0.0195, 0.6223, -0.0134, -0.0253, -0.7822, -0.0020, 0.0020, -0.0078, 99.9000]
	Data: [17200.0000, 0.0078, -0.0156, 0.0000, 0.6223, -0.0134, -0.0253, -0.7823, -0.0020, 0.0020, 0.0000, 99.9000]

Average Data Freq: 164.26 Hz
	Average time step: 6.09 ms
	Min/Max interval: 1.10 / 104.00 ms
	Std Dev: 8.31 ms, jitter: 136.5%

Histogram of report periods (ms):
  0.0–  1.0 (    0) |
  1.0–  2.0 (    4) | #
  2.0–  3.0 (    2) | #
  3.0–  4.0 (   43) | ####
  4.0–  5.0 (  350) | ###################################
  5.0–  6.0 (  496) | ##################################################
  6.0–  7.0 (   61) | ######
  7.0–  8.0 (    4) | #
  8.0–  9.0 (    4) | #
  9.0– 10.0 (    1) | #
 10.0– 11.0 (    4) | #
 11.0– 12.0 (    5) | #
 12.0– 13.0 (    0) |
 13.0– 14.0 (    0) |
 14.0– 15.0 (    0) |
 15.0– 16.0 (    1) | #
 16.0– 17.0 (    1) | #
 17.0– 18.0 (    1) | #
 18.0– 19.0 (    0) |
 19.0– 20.0 (    1) | #
 20.0– 21.0 (    2) | #
     >21.0 (   19) | ##

"""
import binascii
import os
import struct

import numpy as np

# Constants matching Pico code: log_linacc_quat_gyro_flash.py
SECTOR_SIZE = 4096  # Exactly 4 KiB

# DataLog file:# 4096 = 4032 (84 rows * 48 bytes) + 24 bytes of data + 36 null + 4 (CRC)
SECTOR_SIZE = 4096  # Exactly 4 KiB
NUM_FLOATS = 12
BYTES_PER_ROW = 48
ROWS_PER_SECTOR = 84
DATA_SIZE = BYTES_PER_ROW * ROWS_PER_SECTOR  # 4032 bytes = 84 * 48
CUSTOM_DATA_OFFSET = DATA_SIZE
CRC_OFFSET = 4092  # The very last 4 bytes
pack_string = "<" + (NUM_FLOATS * "f")  # number of f's match count

ROW_DTYPE = np.dtype("<f4", NUM_FLOATS)
# Metadata format: < (Little Endian), I (Sector Count), 4f (Triggers/Temp/Time), I (Accuracy)
# Format: <IffffI (Total 24 bytes)
METADATA_FMT = "<IffffI"


def decode_sector_buffer(filename):
    print(f"\n--- Decoding Sector Buffer File (CRC Verified): {filename}")

    rows = []
    corrupt_sector_count = 0

    # Metadata storage lists
    m_start_ts = []
    m_end_ts = []
    m_sector_idx = []
    m_lift_ms = []
    m_flame_ms = []
    m_max_celsius = []
    m_max_celsius_ts = []
    m_min_accuracy = []

    first_timestamp = -1

    with open(filename, "rb") as f:
        sector_idx = 0
        while True:
            sector = f.read(SECTOR_SIZE)
            if not sector:
                break
            if len(sector) < SECTOR_SIZE:
                print(f"Warning: Final sector is incomplete ({len(sector)} bytes). Skipping sector.")
                break

            # CRC is stored at sector end
            data_for_crc = sector[:CRC_OFFSET]
            stored_crc = struct.unpack("<I", sector[CRC_OFFSET:CRC_OFFSET + 4])[0]
            computed_crc = binascii.crc32(data_for_crc) & 0xFFFFFFFF

            if stored_crc != computed_crc:
                print(f"Warning: CRC FAIL! at Sector {sector_idx}: ")
                corrupt_sector_count += 1
                sector_idx += 1
                continue

            # Extract per sector metadata Format: <IffffI
            custom_data = sector[CUSTOM_DATA_OFFSET:CUSTOM_DATA_OFFSET + 24]
            (s_idx,
             lift_ms, flame_ms,
             max_c, max_c_ts,
             min_acc) = struct.unpack("<IffffI", custom_data)

            if s_idx != sector_idx:
                print(f"!!! Sector Index Gap DETECTED: File sector# {sector_idx} vs logged sector# {s_idx}")

            # Append to metadata lists
            m_sector_idx.append(s_idx)
            m_lift_ms.append(lift_ms)
            m_flame_ms.append(flame_ms)
            m_max_celsius.append(max_c)
            m_max_celsius_ts.append(max_c_ts)
            m_min_accuracy.append(min_acc)

            sensor_data = sector[:DATA_SIZE]
            block = np.frombuffer(sensor_data, dtype=ROW_DTYPE).reshape(-1, NUM_FLOATS)

            # Drop all zero rows signal of mid-sector termination, ts_ms will never be 0
            valid = np.any(block != 0.0, axis=1)
            block = block[valid]

            if block.size > 0:
                first_ts = block[0, 0]  # Row 0, Column 0
                last_ts = block[-1, 0]  # Last Row, Column 0
                m_start_ts.append(first_ts)
                m_end_ts.append(last_ts)
                sector_duration_sec = (last_ts - first_ts) / 1000.0
                rows.append(block)
                if first_timestamp == -1:
                    first_timestamp = first_ts
                print(
                    f"Sector {sector_idx}: Duration = {sector_duration_sec:.3f} s, {first_ts:.1f} ms to {last_ts:.1f} ms")
            else:
                m_start_ts.append(0.0)
                m_end_ts.append(0.0)

            sector_idx += 1

        # Convert sensor data to numpy matrix
        data = np.vstack(rows) if rows else np.empty((0, NUM_FLOATS), dtype=np.float32)

        # Convert metadata to numpy vectors
        metadata_vectors = {
            "sector_idx": np.array(m_sector_idx, dtype=np.uint32),
            "start_ts": np.array(m_start_ts, dtype=np.float32),
            "end_ts": np.array(m_end_ts, dtype=np.float32),
            "lift_ms": np.array(m_lift_ms, dtype=np.float32),
            "top_flame_ms": np.array(m_flame_ms, dtype=np.float32),
            "max_celsius": np.array(m_max_celsius, dtype=np.float32),
            "max_celsius_ts": np.array(m_max_celsius_ts, dtype=np.float32),
            "min_accuracy": np.array(m_min_accuracy, dtype=np.uint32)
        }

    print(f"\nDecoded {data.shape[0]} rows from {len(m_sector_idx)} sectors.")
    if corrupt_sector_count > 0:
        print(f"ALERT: {corrupt_sector_count} sectors failed CRC check and were skipped.")

    print(f"Data log time = {(last_ts-first_timestamp)/1000.0:.1f} secs")
    print(f"First timestamp: {first_timestamp:.1f} ms, Last timestamp: {last_ts:.1f} ms\n")

    return data, metadata_vectors, corrupt_sector_count


def ascii_histogram(data):
    """
    Prints horizontal ascii histogram showing times andcounts. Horizontal scale of longest bar is max_width.
    :param data:
    :return:
    """

    # 5ms samples
    # define bins: 0.0–1.0, 1.0–2.0, ..., 9.0–20.0, >20.0
    bin_min = 0.0
    bin_max = 21.0
    bin_width = 1.0

    bins = np.arange(bin_min, bin_max + bin_width, bin_width)
    counts, _ = np.histogram(data, bins=bins)

    # overflow (>10.0)
    overflow = np.sum(data > bin_max)

    # limit histogram length
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

    print(f"      >{bin_max} ({overflow:5d}) | {bar(overflow)}")


def print_summary(data):
    """ print sample of first rows, and print timestamp statistics with ascii histogram"""
    if data.size == 0:
        print("No data")
        return

    h = ["ts_ms", "ax", "ay", "az", "qr", "qi", "qj", "qk", "gy", "gp", "gr", "hpa"]
    print(f"\tHeader: {h}")

    for i in range(min(5, data.shape[0])):
        row = ", ".join(f"{x:.4f}" for x in data[i])
        print(f"\tData: [{row}]")

    time_ms = data[data[:, 0] > 0, 0]
    if time_ms.size < 2:
        print("Not enough data for frequency analysis.")
        return
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


def write_data_csv(filename, data, precision=7):
    """ Write sensor data to CSV"""
    if data.size == 0:
        print(f"No data to write: {filename}")
        return

    header = ["SECONDS", "ax", "ay", "az", "qr", "qi", "qj", "qk", "gy", "gp", "gr", "hpa"]
    fmt = ["%.4f"] + [f"%.{precision}f"] * (data.shape[1] - 1)

    print("\n*** WARNING: CSV is converted to SECONDS, above processing in msec")
    data = np.array(data, copy=True)
    time_ms = data[:, 0].copy()
    data[:, 0] = time_ms / 1000.0

    np.savetxt(filename, data, delimiter=",", header=",".join(header), comments="", fmt=fmt)

    print(f"\nWrote CSV: {filename} ({data.shape[0]} rows)")


def write_metadata_csv(filename, metadata):
    """ Writes per-sector metadata to CSV for flight analysis """
    if not metadata or len(metadata["sector_idx"]) == 0:
        print(f"No metadata to write: {filename}")
        return

    # Convert ms to seconds for consistency with your sensor data CSV
    start_sec = metadata["start_ts"] / 1000.0
    end_sec = metadata["end_ts"] / 1000.0
    duration = end_sec - start_sec
    lift_sec = metadata["lift_ms"] / 1000.0
    flame_sec = metadata["top_flame_ms"] / 1000.0
    temp_ts_sec = metadata["max_celsius_ts"] / 1000.0

    # Stack columns horizontally
    # We use metadata["sector_idx"] and metadata["min_accuracy"] as floats for stacking,
    # but we can format them as ints in the fmt string
    meta_stack = np.column_stack((
        metadata["sector_idx"],
        start_sec,
        end_sec,
        duration,
        lift_sec,
        flame_sec,
        metadata["max_celsius"],
        temp_ts_sec,
        metadata["min_accuracy"]
    ))

    # Metadata Format
    header = [
        "sector_idx", "start_sec", "end_sec", "duration_sec",
        "lift_sec", "flame_sec", "max_celsius", "max_celsius_ts_sec", "min_accuracy"
    ]
    fmt = ["%d", "%.4f", "%.4f", "%.4f", "%.4f", "%.4f", "%.2f", "%.4f", "%d"]

    np.savetxt(filename, meta_stack, delimiter=",", header=",".join(header), comments="", fmt=fmt)

    print(f"Wrote Metadata CSV: {filename} ({meta_stack.shape[0]} sectors)")


# Main #############################################
def main():
    # Convert Sector-formatted file
    filename = "imu_test_data_logs/flight_log_debug_sector.bin"

    if os.path.exists(filename):
        sensor_data, sensor_metadata, corrupt_sector_count = decode_sector_buffer(filename)

        # print flight sensor stats
        print_summary(sensor_data)

        # Print metadata stats

        overall_max_c = np.max(sensor_metadata['max_celsius'])
        overall_min_c = np.min(sensor_metadata['max_celsius'])
        max_temp_idx = np.argmax(sensor_metadata['max_celsius'])
        max_temp_ts = sensor_metadata['max_celsius_ts'][max_temp_idx]

        # Fusible Trigger Signal Analysis
        def analyze_trigger(vector):
            # Get only the non-zero unique values
            hits = vector[vector > 0]
            unique_hits = np.unique(hits)

            val = unique_hits[0] if unique_hits.size > 0 else 0.0
            status = ""
            if unique_hits.size > 1:
                status = f"!! WARNING: {unique_hits.size} multiple fusible signals !!"
            elif unique_hits.size == 0:
                status = "Not Detected"

            return val, status

        lift_val, lift_status = analyze_trigger(sensor_metadata['lift_ms'])
        flame_val, flame_status = analyze_trigger(sensor_metadata['top_flame_ms'])

        sequence_warning = ""
        time_delta_ms = 0.0

        if lift_val > 0 and flame_val > 0:
            time_delta_ms = flame_val - lift_val
            if time_delta_ms < 0:
                sequence_warning = "!! SEQUENCE ERROR: Flame detected BEFORE Lift !!"
            else:
                sequence_warning = f"Interval: {time_delta_ms:.2f} ms"
        elif lift_val > 0 or flame_val > 0:
            sequence_warning = "Interval: N/A (Missing one trigger)"
        else:
            sequence_warning = "Interval: N/A (No triggers)"

        # Accuracy and Sector Stats
        max_acc = np.max(sensor_metadata['min_accuracy'])
        min_acc = np.min(sensor_metadata['min_accuracy'])
        num_sectors = len(sensor_metadata['sector_idx'])

        # --- PRINT RESULTS ---
        print(f"\nFLIGHT METADATA SUMMARY ({num_sectors} Sectors)")
        print(f"Lift Time:      {lift_val:10.3f} ms | {lift_status}")
        print(f"Top Flame Time: {flame_val:10.3f} ms | {flame_status}")
        print(f"Lift to Top Flame:   {time_delta_ms:6.4f} ms| {flame_status}")
        print(f"Fusible Events:      {sequence_warning}")
        print(f"\nMax Celsius:    {overall_max_c:.2f}° C at {max_temp_ts:.1f} ms")
        print(f"Min Celsius:    {overall_min_c:.2f}° C")
        print(f"Accuracy Range: Min: {min_acc} | Max: {max_acc}")

        # create sensor csv and metadata csv
        write_data_csv(filename.replace(".bin", ".csv"), sensor_data)
        write_metadata_csv(filename.replace(".bin", "_metadata.csv"), sensor_metadata)

    else:
        print(f"\nError: File {filename} does not exist")


if __name__ == "__main__":
    main()
