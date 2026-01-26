# read_prepare_9_dof.py

"""
Linear_acceleration, Quaternion, Gyro logging from BNO086 sensor to flash memory.

This code checks the calibration accuracy of the bno.linear_acceleration, bno.quaternion, and bno.gyro and
continues this until all have a better accuracy >=2 (medium to high). Then saves the calibration to the
BNO086.

After calibration, the bno.linear_acceleration and bno.gyro are measured while not moving to create
a bias correction. The values collected are shown in an ascii histogram. Then the median is used
as the bias correction that is applied to dall data.

There are two methods included for storing the data to flash:
    1. OPTION 1: Buffer all data in memory before writing to Flash - limited to 95 KiB logs
        write_results_whole_batch(bno, rows, filename)
    The max size of 95 KiB is a serious limitation due to typical sensor result volumes.
    This method is included for TESTING-ONLY, as it is low-jitter

    2. OPTION 2: Buffer 4 KiB Sector, then sector to flash. This will show jitter at sector writes.
        write_results_by_sector(bno, rows, filename)
    The max size of storage is limited only by the free space on flash. It gathers sensor result <- 4KiB.
    The 4 KiB limit is due to the Flash's Sector size of 4 KiB. Writing this size is the most efficient.
    With 5ms sample rate, the flash write and intermittant flush (duty cycle settable in code) will cause
    50 ms to 110 ms jitter in sample collection.

The BNO086 sensor is connect to Raspberry Pi Pico 2 W by SPI.

Input:
    *** CAUTION: TIME IN msec NOT SECONDS, for BNO086 efficiency at 5ms 200Hz
    ax, ay, az, acc, ts_ms = bno.linear_acceleration.full
    qr, qi, qj, qk = bno.quaternion
    gy, gp, gr = bno.gyro

Output:
    *** CAUTION: The created CSV IS CONVERTED TO SECONDS by unpack_bin_sensor_logs.py
    Seconds, lin_acc_x, lin_acc_y, lin_acc_z, quat_r, quat_i, quat_j, quat_k, gyro_y, gyro_p gyro_r

# Reading Linear_Acc for 1000 rows, doing nothing with output
# sensor timestamps last_sensor_ms=5724.8 first_sensor_ms=644.9  sensor duration: 5.1 s
# Sensor msec/Lin_Acc = 5.08 ms
# Clock msec/Lin_Acc  = 5.06 ms
#
# Writing data in sector chunks to flash in 4 KiB sectors with flush no CRC
# Array each result for 1000 rows:
# sensor timestamps last_sensor_ms=11931.5 first_sensor_ms=5724.8  sensor duration: 6.2 s
# Sensor msec/reports = 6.21 ms
# Clock msec/reports  = 6.28 ms
# BYTES_PER_ROW=44, data size = 44000 bytes
# Array = 43.0 KiB, xfer = 6.8 KiB/s
#
# Writing data in sector chunks to flash in 4 KiB sectors with flush WITH CRC
# Array each result for 1000 rows:
# sensor timestamps last_sensor_ms=11997.0 first_sensor_ms=5785.0  sensor duration: 6.2 s
# Sensor msec/reports = 6.21 ms
# Clock msec/reports  = 6.29 ms
# BYTES_PER_ROW=44, data size = 44000 bytes
# Array = 43.0 KiB, xfer = 6.8 KiB/s
"""

import binascii  # For fast CRC32
import gc
import os
import struct

from array import array
from bno08x import *
from machine import SPI, Pin
from spi import BNO08X_SPI
from utime import sleep_ms, ticks_ms, sleep_us

NUM_FLOATS = const(11)
BYTES_PER_ROW = const(44)
pack_string = "<" + (NUM_FLOATS * "f")  # number of f's match count

# CAUTION: When change NUM_FLOATS, need to change these constants
ROWS_PER_SECTOR = const(93)
DATA_SIZE = BYTES_PER_ROW * ROWS_PER_SECTOR  # 4092 = 44 * 93
SECTOR_SIZE = const(4096)  # Exactly 4 KiB

# GLOBALS for Bias Correction
AX_BIAS = 0.0
AY_BIAS = 0.0
AZ_BIAS = 0.0
GY_BIAS = 0.0
GP_BIAS = 0.0
GR_BIAS = 0.0


def write_results_whole_batch(bno, rows: int, filename: str):
    """
    Buffer all results in memory then write whole file to flash. The advantage is very little jitter
    for high-frequency updates (5 ms) with little jitter.

    Low jitter, but limited to only 95 KiB in-memory storage.
    This function usefor for simple tests, typically use sector-format version.

    :param bno:
    :param rows:
    :param filename:
    :return:
    """
    print("\nStore in buffer, then write whole file")

    # Create/Clear the file
    with open(filename, "wb") as f:
        pass

    # create In-memory buffer, note fragmented heap may limit size
    required_buffer_size = rows * BYTES_PER_ROW
    free_heap_size = gc.mem_free()

    # Conservative margin: fragmentation, stack, and other
    SAFETY_MARGIN = 16 * 1024  # 16 KiB

    if required_buffer_size > (free_heap_size - SAFETY_MARGIN):
        raise MemoryError(
            "Whole-batch buffer too large: "
            f"need {required_buffer_size / 1024.0:.1f} KiB, have {(free_heap_size - SAFETY_MARGIN) / 1024.0:.1f} KiB free. "
            f"Max rows={(free_heap_size - SAFETY_MARGIN) / BYTES_PER_ROW:.0f} "
            "Use sector-based logging instead."
        )
    else:
        percent_memory = (required_buffer_size / free_heap_size) * 100.0
        print(f"Memory used for buffer: {percent_memory:.0f}%")

    buffer = bytearray(required_buffer_size)

    # localize globals for effiency
    ax_offset, ay_offset, az_offset = AX_BIAS, AY_BIAS, AZ_BIAS
    gy_offset, gp_offset, gr_offset = GY_BIAS, GP_BIAS, GR_BIAS

    update = bno.update_sensors
    pack_into = struct.pack_into
    crc32 = binascii.crc32
    lin_acc = bno.linear_acceleration
    quat = bno.quaternion
    gyro = bno.gyro

    i = 0
    start = ticks_ms()
    while i < rows:
        if not update():
            continue

        if bno.linear_acceleration.updated:
            ax, ay, az, acc, ts_ms = lin_acc.full
            qr, qi, qj, qk = quat
            gy, gp, gr = gyro

            if i == 0:
                first_sensor_ms = ts_ms

            offset = i * BYTES_PER_ROW
            pack_into(pack_string, buffer, offset,
                      ts_ms,
                      ax - ax_offset, ay - ay_offset, az - az_offset,
                      qr, qi, qj, qk,
                      gy - gy_offset, gp - gp_offset, gr - gr_offset, )

            i += 1

    last_sensor_ms = ts_ms
    pico_ms = ticks_diff(ticks_ms(), start)

    print(f"\nPrinting each Linear_Acc for {rows} rows:")
    print(
        f"sensor timestamps {last_sensor_ms=} {first_sensor_ms=}  sensor duration: {(last_sensor_ms - first_sensor_ms) / 1000:.1f} s")
    print(f"Sensor msec/reports = {(last_sensor_ms - first_sensor_ms) / rows:.2f} ms")

    print(f"Clock msec/reports  = {(pico_ms / rows):.2f} ms")

    write_start = ticks_ms()
    with open(filename, "ab") as f:
        f.write(buffer)  # Write whole buffer 44,000 bytes
        f.flush()
        os.sync()

    kbytes = len(buffer) / 1024.0
    secs = ticks_diff(ticks_ms(), write_start) / 1000.0
    print(f"Time to write {kbytes} KiB, time = {secs} s, xfer = {(kbytes / secs):.1f} KiB/s")


def write_results_by_sector(bno, rows: int, filename: str):
    """
    Write results to file sector by sector. At high-frequency (5ms updates) For 11 floats on BNO086,
    this is about 1 second of data which minimizes loss when sensor in hostile environment.
    Sectors are the most efficient write size.

    The disadvantage is about 50-70ms delay at sector write, and > 100ms at flush which will increase
    the jitter at sector boundaries.

    :param bno:
    :param rows:
    :param filename:
    :return:
    """
    global AX_BIAS, AY_BIAS, AZ_BIAS, GY_BIAS, GP_BIAS, GR_BIAS

    # Time Packing data into a 4 KiB buffer & writing sectors for flash
    print("\nWriting data to Flash in 4 KiB sector chunks to flash")

    # Reset file in sector data format
    with open(filename, "wb") as f:
        pass

    # Buffer of exactly 4 KiB, data: 4092 CRC: last 4 bytes
    sector_buffer = bytearray(SECTOR_SIZE)

    # localize globals for effiency
    ax_offset, ay_offset, az_offset = AX_BIAS, AY_BIAS, AZ_BIAS
    gy_offset, gp_offset, gr_offset = GY_BIAS, GP_BIAS, GR_BIAS

    update = bno.update_sensors
    pack_into = struct.pack_into
    crc32 = binascii.crc32
    lin_acc = bno.linear_acceleration
    quat = bno.quaternion
    gyro = bno.gyro

    i = 0
    sector_count = 0
    start = ticks_ms()
    with open(filename, "ab") as f:
        while i < rows:

            # Accumulate one sector of data ~ 96 rows
            sector_row_count = 0
            while sector_row_count < ROWS_PER_SECTOR and i < rows:
                if not update():
                    continue

                if lin_acc.updated:
                    ax, ay, az, acc, ts_ms = lin_acc.full
                    qr, qi, qj, qk = quat
                    gy, gp, gr = gyro

                    if i == 0:
                        first_sensor_ms = ts_ms

                    # Pack ONLY into the sector buffer
                    offset = sector_row_count * BYTES_PER_ROW
                    pack_into(pack_string, sector_buffer, offset,
                              ts_ms,
                              ax - ax_offset, ay - ay_offset, az - az_offset,
                              qr, qi, qj, qk,
                              gy - gy_offset, gp - gp_offset, gr - gr_offset, )

                    sector_row_count += 1
                    i += 1

            # ZERO-FILL unused rows
            if sector_row_count < ROWS_PER_SECTOR:
                start_fill = sector_row_count * BYTES_PER_ROW
                sector_buffer[start_fill:DATA_SIZE] = b"\x00" * (DATA_SIZE - start_fill)

            # Calculate CRC32 on the data (first 4092 bytes), Pack the CRC  32bit "I" into the last 4 sector bytes
            # adds .08 ms to loop  6.18 ms with flush, 6.26 with flush & CRC
            crc = crc32(memoryview(sector_buffer)[:DATA_SIZE])
            struct.pack_into("<I", sector_buffer, DATA_SIZE, crc)

            # Write sector to flash:  bytes 0-4091 are data, last 4 bytes are CRC or 0x00 padding
            # write_start = ticks_ms()
            f.write(sector_buffer)  # Write exactly 4 KiB

            # Flush every 2*93 rows (about 1 sec)
            if sector_count % 2 == 0:
                f.flush()

            sector_count += 1

            # Debug timing for each write, measured 45 ms
            # write_time = ticks_diff(ticks_ms(), write_start)
            # print(f"Sector flushed (4 KiB). Write: {write_time} ms. Total Rows so far: {i}")

        f.flush()
        os.sync()

    last_sensor_ms = ts_ms
    pico_ms = ticks_diff(ticks_ms(), start)

    print(f"Final flush and sync. Total rows: {i} Rows")
    print(
        f"Sensor timestamps {last_sensor_ms=} {first_sensor_ms=}  sensor duration: {(last_sensor_ms - first_sensor_ms) / 1000:.1f} s")
    print(f"Sensor msec/reports = {(last_sensor_ms - first_sensor_ms) / rows:.2f} ms")

    print(f"Clock msec/reports  = {(pico_ms / rows):.2f} ms")

    print(f"{BYTES_PER_ROW=}, data size = {(rows * BYTES_PER_ROW)} bytes")
    kbytes = (BYTES_PER_ROW * rows) / 1024
    print(f"Array = {kbytes:.1f} KiB, xfer = {kbytes / (pico_ms / 1000.0):.1f} KiB/s")


def ascii_histogram(data, bins=15, max_width=40):
    """
    Automatically bins data based on min/max and prints an ASCII histogram.
    Works with numpy arrays or standard lists.
    """
    if len(data) == 0:
        print("No data to histogram.")
        return

    d_min, d_max = min(data), max(data)

    # If all data is identical (e.g., all 0.0), create a small range to avoid division by zero
    if d_min == d_max:
        d_max += 0.1

    # Calculate bin edges and counts
    bin_width = (d_max - d_min) / bins
    counts = [0] * bins

    for val in data:
        # Calculate which bin the value belongs to
        idx = int((val - d_min) / bin_width)
        if idx == bins: # Handle maximum value
            idx -= 1
        counts[idx] += 1

    # Scale bars
    max_count = max(counts)
    scale = max_width / max_count if max_count > 0 else 1.0

    print(f"\nHistogram (Range: {d_min:.6f} to {d_max:.6f})")
    print("-" * (max_width + 35))

    for i in range(bins):
        b_start = d_min + (i * bin_width)
        b_end = b_start + bin_width

        bar_len = int(counts[i] * scale)
        bar = "#" * bar_len

        # Print row: Range | Count | Bar
        print(f"{b_start:10.6f} to {b_end:10.6f} | {counts[i]:5d} | {bar}")

def get_median(data):
    """Simple median for array.array or list"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2


def get_static_bias(bno, samples=100):
    """
    Calculate residual bias whilee sensor stable. Do this for number of 'samples'

    EXAMPLE OUTPUT
    --------------
    ascii Histograms of Acceleration Biases (m/s²):

    bias will use median(ax) = -0.0039063

    Histogram (Range: -0.085938 to 0.722656)
    ---------------------------------------------------------------------------
     -0.085938 to  -0.032031 |     1 |
     -0.032031 to   0.021875 |    94 | ########################################
      0.021875 to   0.075781 |     1 |
      0.075781 to   0.129688 |     0 |
      0.129688 to   0.183594 |     0 |
      0.183594 to   0.237500 |     0 |
      0.237500 to   0.291406 |     0 |
      0.291406 to   0.345312 |     0 |
      0.345312 to   0.399219 |     0 |
      0.399219 to   0.453125 |     0 |
      0.453125 to   0.507031 |     0 |
      0.507031 to   0.560938 |     0 |
      0.560938 to   0.614844 |     0 |
      0.614844 to   0.668750 |     0 |
      0.668750 to   0.722656 |     4 | #

    bias will use median(ay) = 0.0000000

    Histogram (Range: -9.101562 to 0.164062)
    ---------------------------------------------------------------------------
     -9.101562 to  -8.483854 |     4 | #
     -8.483854 to  -7.866146 |     0 |
     -7.866146 to  -7.248438 |     0 |
     -7.248438 to  -6.630729 |     0 |
     -6.630729 to  -6.013021 |     0 |
     -6.013021 to  -5.395313 |     0 |
     -5.395312 to  -4.777604 |     0 |
     -4.777604 to  -4.159896 |     0 |
     -4.159896 to  -3.542188 |     0 |
     -3.542188 to  -2.924480 |     0 |
     -2.924480 to  -2.306771 |     0 |
     -2.306771 to  -1.689062 |     0 |
     -1.689063 to  -1.071354 |     0 |
     -1.071354 to  -0.453646 |     0 |
     -0.453646 to   0.164063 |    96 | ########################################

    bias will use median(az) = 0.0000000

    Histogram (Range: -0.093750 to 7.046875)
    ---------------------------------------------------------------------------
     -0.093750 to   0.382292 |    96 | ########################################
      0.382292 to   0.858333 |     0 |
      0.858333 to   1.334375 |     0 |
      1.334375 to   1.810417 |     0 |
      1.810417 to   2.286458 |     0 |
      2.286458 to   2.762500 |     0 |
      2.762500 to   3.238542 |     0 |
      3.238542 to   3.714583 |     0 |
      3.714583 to   4.190625 |     0 |
      4.190625 to   4.666667 |     0 |
      4.666667 to   5.142709 |     0 |
      5.142708 to   5.618750 |     0 |
      5.618750 to   6.094792 |     0 |
      6.094792 to   6.570834 |     0 |
      6.570833 to   7.046875 |     4 | #

    ascii Histograms of Gyroscope Biases (rad/s):

    bias will use median(gy) = 0.0000000

    Histogram (Range: -0.810547 to 0.013672)
    ---------------------------------------------------------------------------
     -0.810547 to  -0.755599 |     2 |
     -0.755599 to  -0.700651 |     0 |
     -0.700651 to  -0.645703 |     0 |
     -0.645703 to  -0.590755 |     0 |
     -0.590755 to  -0.535807 |     0 |
     -0.535807 to  -0.480859 |     0 |
     -0.480859 to  -0.425911 |     0 |
     -0.425911 to  -0.370964 |     0 |
     -0.370964 to  -0.316016 |     0 |
     -0.316016 to  -0.261068 |     0 |
     -0.261068 to  -0.206120 |     0 |
     -0.206120 to  -0.151172 |     0 |
     -0.151172 to  -0.096224 |     0 |
     -0.096224 to  -0.041276 |     0 |
     -0.041276 to   0.013672 |    98 | ########################################

    bias will use median(gp) = 0.0000000

    Histogram (Range: -0.363281 to 0.005859)
    ---------------------------------------------------------------------------
     -0.363281 to  -0.338672 |     2 |
     -0.338672 to  -0.314062 |     0 |
     -0.314062 to  -0.289453 |     0 |
     -0.289453 to  -0.264844 |     0 |
     -0.264844 to  -0.240234 |     0 |
     -0.240234 to  -0.215625 |     0 |
     -0.215625 to  -0.191016 |     0 |
     -0.191016 to  -0.166406 |     0 |
     -0.166406 to  -0.141797 |     0 |
     -0.141797 to  -0.117188 |     0 |
     -0.117188 to  -0.092578 |     0 |
     -0.092578 to  -0.067969 |     0 |
     -0.067969 to  -0.043359 |     0 |
     -0.043359 to  -0.018750 |     0 |
     -0.018750 to   0.005859 |    98 | ########################################

    bias will use median(gr) = 0.0000000

    Histogram (Range: -0.003906 to 0.162109)
    ---------------------------------------------------------------------------
     -0.003906 to   0.007161 |    98 | ########################################
      0.007161 to   0.018229 |     0 |
      0.018229 to   0.029297 |     0 |
      0.029297 to   0.040365 |     0 |
      0.040365 to   0.051432 |     0 |
      0.051432 to   0.062500 |     0 |
      0.062500 to   0.073568 |     0 |
      0.073568 to   0.084635 |     0 |
      0.084635 to   0.095703 |     0 |
      0.095703 to   0.106771 |     0 |
      0.106771 to   0.117839 |     0 |
      0.117839 to   0.128906 |     0 |
      0.128906 to   0.139974 |     0 |
      0.139974 to   0.151042 |     0 |
      0.151042 to   0.162109 |     2 |

    Static Acceleration Biases: AX_BIAS=-0.003906, AY_BIAS=+0.000000, AZ_BIAS=+0.000000
    Static Gyro Biases:         GY_BIAS=+0.000000, GP_BIAS=+0.000000, GR_BIAS=+0.000000

    """
    global AX_BIAS, AY_BIAS, AZ_BIAS, GY_BIAS, GP_BIAS, GR_BIAS

    print(f"Calculating static bias from {samples} samples.")
    print("\n* DO NOT MOVE SENSOR...")
    sum_ax, sum_ay, sum_az = 0.0, 0.0, 0.0
    sum_gy, sum_gp, sum_gr = 0.0, 0.0, 0.0

    ax = array('f', [0.0] * samples)
    ay = array('f', [0.0] * samples)
    az = array('f', [0.0] * samples)
    gy = array('f', [0.0] * samples)
    gp = array('f', [0.0] * samples)
    gr = array('f', [0.0] * samples)

    idx = 0
    while idx < samples:
        bno.update_sensors()
        if bno.linear_acceleration.updated:

            new_ax, new_ay, new_az = bno.linear_acceleration
            new_gy, new_gp, new_gr = bno.gyro

            # Store in float arrays
            ax[idx], ay[idx], az[idx] = new_ax, new_ay, new_az
            gy[idx], gp[idx], gr[idx] = new_gy, new_gp, new_gr

            idx += 1

    # There can be significant outliers, using median instead of average
    AX_BIAS = get_median(ax)
    AY_BIAS = get_median(ay)
    AZ_BIAS = get_median(az)
    GY_BIAS = get_median(gy)
    GP_BIAS = get_median(gp)
    GR_BIAS = get_median(gr)

    print("\nascii Histograms of Acceleration Biases (m/s²):")
    print(f"\nbias will use median(ax) = {AX_BIAS:.7f}")
    ascii_histogram(ax, bins=15)
    print(f"\nbias will use median(ay) = {AY_BIAS:.7f}")
    ascii_histogram(ay, bins=15)
    print(f"\nbias will use median(az) = {AZ_BIAS:.7f}")
    ascii_histogram(az, bins=15)

    print("\nascii Histograms of Gyroscope Biases (rad/s):")
    print(f"\nbias will use median(gy) = {GY_BIAS:.7f}")
    ascii_histogram(gy, bins=15)
    print(f"\nbias will use median(gp) = {GP_BIAS:.7f}")
    ascii_histogram(gp, bins=15)
    print(f"\nbias will use median(gr) = {GP_BIAS:.7f}")
    ascii_histogram(gr, bins=15)


def sensor_calibration(bno, stable_sec):
    """ Sensor calibration, must be stable for stable_sec. TODO no max for timeout
    :param bno:
    :param stable_sec:
    :return:
    """
    print(f"\nCalibration: Continue for {stable_sec} secs of Medium(2) to High(3) Accuracy\n")
    start_good = None
    calibration_good = False
    status = ""

    # Begin calibration, Wait sensor to be ready to calibrate
    bno.begin_calibration()
    bno.calibration_status()

    last_print = ticks_ms()
    while True:
        bno.update_sensors()

        # only print every .2 sec (200 ms)
        if ticks_diff(ticks_ms(), last_print) < 200:
            continue
        last_print = ticks_ms()

        _, _, _, accel_accuracy, _ = bno.linear_acceleration.full
        _, _, _, gyro_accuracy, _ = bno.gyro.full
        _, _, _, _, quat_accuracy, _ = bno.quaternion.full

        if all(x >= 2 for x in (accel_accuracy, gyro_accuracy, quat_accuracy)):
            status = "All Sensors >= 2"
            calibration_good = True
        else:
            if start_good is not None:
                print("\nlost calibration, resetting timer\n")
            status = "low accuracy, suggest moving sensor"
            calibration_good = False

        print(f"Accuracy: accel={accel_accuracy}, gyro={gyro_accuracy}, quat={quat_accuracy}\t{status}")

        if calibration_good:
            if start_good is None:
                start_good = ticks_ms()
                print(f"Calibration >=2 on all sensors. Start {stable_sec}-second timer...\n")
            else:
                elapsed = ticks_diff(ticks_ms(), start_good) / 1000.0
                if elapsed >= stable_sec:
                    print(f"*** Calibration stable for {stable_sec} secs")
                    break
        else:
            start_good = None

    bno.save_calibration_data()
    print("*** Calibration saved")


def main():
    int_pin = Pin(14, Pin.IN)  # Interrupt, enables BNO to signal when ready
    reset_pin = Pin(15, Pin.OUT, value=1)  # Reset to signal BNO to reset

    # miso=Pin(16) - BNO SO (POCI)
    cs_pin = Pin(17, Pin.OUT, value=1)
    # sck=Pin(18)  - BNO SCK
    # mosi=Pin(19) - BNO SI (PICO)
    wake_pin = Pin(20, Pin.OUT, value=1)  # BNO WAK

    spi = SPI(0, baudrate=3000000, sck=Pin(18), mosi=Pin(19), miso=Pin(16))
    bno = BNO08X_SPI(spi, cs_pin, reset_pin, int_pin, wake_pin, debug=False)

    print(spi)  # baudrate=3000000 required
    print("Start")
    print("====================================\n")

    # Update frequency in Hz, 200Hz = 5ms sample
    # very slow for oriention testing: 10Hz = 100ms
    update_frequency = 200
    bno.linear_acceleration.enable(update_frequency)
    bno.gyro.enable(update_frequency)
    bno.quaternion.enable(update_frequency)

    bno.print_report_period()
    print("BNO08x sensors enabled\n")

    # Check Calibration
    # CAUTION: this will NOT time out if inaccurate
    sensor_calibration(bno, stable_sec=3)

    print("\nSleeping for 5 sec - Make sure sensor still for Bias calibration")
    sleep_ms(5000)

    # Calculate Sensor Bias DO Not move sensor
    print("\nStarting Bias calibration...")
    get_static_bias(bno, samples=10 * update_frequency)
    print(f"\nStatic Acceleration Biases: {AX_BIAS=:+.6f}, {AY_BIAS=:+.6f}, {AZ_BIAS=:+.6f}")
    print(f"Static Gyro Biases:         {GY_BIAS=:+.6f}, {GP_BIAS=:+.6f}, {GR_BIAS=:+.6f}")

    # Log results

    # 5ms sample period generate 200 rows/sec
    duration_seconds = 20
    rows = duration_seconds * update_frequency
    print(f"\nSensor Collection: {duration_seconds=}, {rows=}, {update_frequency=}Hz,")

    # --- TWO WRITE-TO-FLASH Options - TODO CHOOSE ONE

    # OPTION 1: Buffer all data in memory before writing to Flash - limited to 95 KiB logs
    filename = "flight_log_2026xxxx_xpm_whole.bin"
    write_results_whole_batch(bno, rows, filename)

    # OPTION 2: Buffer 4 KiB Sector, then sector to flash. This will show jitter at sector writes.
    filename = "flight_log_2026xxxx_xpm_sector.bin"
    write_results_by_sector(bno, rows, filename)


if __name__ == "__main__":
    main()
