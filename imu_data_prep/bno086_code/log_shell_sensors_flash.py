# log_shell_sensors_flash.py
"""
Linear_acceleration, Quaternion, Gyro logging from BNO086 sensor to flash memory.

This code checks the calibration accuracy of the bno.linear_acceleration, bno.quaternion, and bno.gyro and
continues this until all have a better accuracy >=2 (medium to high). Then saves the calibration to the
BNO086.

After calibration, the bno.linear_acceleration and bno.gyro are measured while not moving to create
a bias correction. The values collected are shown in an ascii histogram. Then the median is used
as the bias correction that is applied to dall data.

Beause of the harsh enviroment which make disable the sensor. This code buffers data in memory and then
writes it to flash. To be efficent for the flash it Buffers in 4 KiB Sector-size, then writes the
sector to flash. This will show jitter at sector writes.
        write_results_by_sector(bno, rows, filename)

The max size of storage is limited only by the free space on flash. It gathers sensor result <- 4KiB.
The 4 KiB limit is due to the Flash's Sector size of 4 KiB. Writing this size is the most efficient.
With 5ms sample rate, the flash write and intermittant flush (duty cycle settable in code) will cause
50 ms to 110 ms jitter in sample collection. Also at 5ms (200Hz) a sector-size is about 0.5sec of data.

The BNO086 sensor is connected to Raspberry Pi Pico 2 W by SPI.

Input:
    *** CAUTION: TIME IN msec NOT SECONDS, for BNO086 efficiency at 5ms 200Hz
    ax, ay, az, acc, ts_ms = bno.linear_acceleration.full
    qr, qi, qj, qk = bno.quaternion
    gy, gp, gr = bno.gyro
    hpa = bmp.pressure

Output:
    *** CAUTION: The created CSV IS CONVERTED TO SECONDS by unpack_bin_sensor_logs.py
    Seconds, lin_acc_x, lin_acc_y, lin_acc_z, quat_r, quat_i, quat_j, quat_k, gyro_y, gyro_p, gyro_r, hpa

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

Previous Version:
    The previous version of this code log_linacc_quat_gyro_flash_spi.py also had debug
    option to buffer all data in heap memory but it was limited to a max of 95 KiB logs.
        write_results_whole_batch(bno, rows, sensor_file_name)
"""

import binascii  # For fast CRC32
import gc
import os
import struct
from array import array

from bno08x import *
from machine import SPI, I2C, Pin
from micropython_bmpxxx import bmpxxx
from spi import BNO08X_SPI
from utime import sleep_ms, sleep_us, ticks_ms, ticks_us, ticks_add

# File
SENSOR_FILE_NAME = "flight_log_debug_sector_4.bin"

# Site Constants
PDX_HOME = 104.851       # peakfinder: 45.49720,-122.74612
SARA_B_LINE_WPA_AZ = 299.923 # peakfinder: 34.444574,-114.250271
SITE_ELEVATION = SARA_B_LINE_WPA_AZ
# https://www.weather.gov/wrh/timeseries?site=KHII
# Elev: 807.0 ft; Lat/Lon: 34.56595/-114.35224
# 
# Nearest airport: Kingman Airport (70 miles away), KIGM
# https://www.weather.gov/wrh/timeseries?site=KIGM
# Elev: 3445.0 ft; Lat/Lon: 35.25778/-113.93306


# DataLog file:# 4096 = 4032 (84 rows * 48 bytes) + 24 bytes of data + 36 null + 4 (CRC)
SECTOR_SIZE = const(4096)  # Exactly 4 KiB
NUM_FLOATS = const(12)
BYTES_PER_ROW = const(48)
ROWS_PER_SECTOR = const(84)
DATA_SIZE = BYTES_PER_ROW * ROWS_PER_SECTOR  # 4032 bytes = 84 * 48
CUSTOM_DATA_OFFSET = DATA_SIZE
CRC_OFFSET = const(4092)  # The very last 4 bytes
pack_string = "<" + (NUM_FLOATS * "f")  # number of f's match count

# GLOBALS for Bias Correction
AX_BIAS = 0.0
AY_BIAS = 0.0
AZ_BIAS = 0.0
GY_BIAS = 0.0
GP_BIAS = 0.0
GR_BIAS = 0.0

# Get synch time, usec start point relative to the msec using ticks_diff
_sync_ms = ticks_ms()
_sync_us = ticks_us()

def get_synced_ms(target_us):
    """ Get microsecond value (target_us) and correlates with ticks_ms()"""
    # Add the fractional milliseconds to anchor millisecond timestamp
    delta_us = ticks_diff(target_us, _sync_us)
    return _sync_ms + (delta_us / 1000.0)


# Global timestamps for interrupts
lift_trigger_us = 0
top_flame_trigger_us = 0


def handle_lift_interrupt(pin):
    global lift_trigger_us
    lift_trigger_us = ticks_us()
    # print(f"Lift interrupt: {lift_trigger_us=}")
    pin.irq(handler=None)


def handle_top_flame(pin):
    global top_flame_trigger_us
    top_flame_trigger_us = ticks_us()
    # print(f"Top flame interrupt: {top_flame_trigger_us=}")
    pin.irq(handler=None)


def write_results_by_sector(bno, bmp, rows: int, sensor_file_name: str):
    """
    Write results to file sector by sector. At high-frequency (5ms updates) For 11 floats on BNO086,
    this is about 1 second of data which minimizes loss when sensor in hostile environment.
    Sectors are the most efficient write size.

    The disadvantage is about 50-70ms delay at sector write, and > 100ms at flush which will increase
    the jitter at sector boundaries.
    """
    global AX_BIAS, AY_BIAS, AZ_BIAS, GY_BIAS, GP_BIAS, GR_BIAS

    lift_trigger_ms = 0
    top_flame_trigger_ms = 0

    # Time Packing data into a 4 KiB buffer & writing sectors for flash
    print("\nWriting data to Flash in 4 KiB sectors to flash")

    # Reset file in sector data format
    with open(sensor_file_name, "wb") as f:
        pass

    # Buffer of exactly 4 KiB, data: 4092 CRC: last 4 bytes
    sector_buffer = bytearray(SECTOR_SIZE)

    # localize globals for efficiency
    ax_offset, ay_offset, az_offset = AX_BIAS, AY_BIAS, AZ_BIAS
    gy_offset, gp_offset, gr_offset = GY_BIAS, GP_BIAS, GR_BIAS

    update = bno.update_sensors
    pack_into = struct.pack_into
    crc32 = binascii.crc32
    lin_acc = bno.linear_acceleration
    quat = bno.quaternion
    gyro = bno.gyro

    i = 0
    sector_idx = 0
    start = ticks_ms()
    _, _, _, min_accuracy, ts_ms = lin_acc.full
    first_sensor_ms = ts_ms

    with open(sensor_file_name, "ab") as f:
        print(f"Sensor Data file: {sensor_file_name}")

        lift_trigger_ms = 0.0
        top_flame_trigger_ms = 0.0
        while i < rows:

            # Start New Sector Write: 4096 sector-sized batches
            sector_row_count = 0
            max_celsius_during_sector = bmp.temperature
            max_celsius_ts_ms = ts_ms
            min_accuracy = 9

            # zero-fill metadata & CRC part of sector buffer
            sector_buffer[DATA_SIZE:] = b"\x00" * (SECTOR_SIZE - DATA_SIZE)

            while sector_row_count < ROWS_PER_SECTOR and i < rows:
                if not update():
                    continue

                if lin_acc.updated:
                    ax, ay, az, acc, ts_ms = lin_acc.full
                    qr, qi, qj, qk = quat
                    gy, gp, gr = gyro
                    hpa = bmp.pressure
                    celsius = bmp.temperature

                    if celsius > max_celsius_during_sector:
                        max_celsius_during_sector = celsius
                        max_celsius_ts_ms = ts_ms

                    min_accuracy = min(min_accuracy, acc)

                    # Pack ONLY into the sector buffer
                    offset = sector_row_count * BYTES_PER_ROW
                    pack_into(pack_string, sector_buffer, offset,
                              ts_ms,
                              ax - ax_offset, ay - ay_offset, az - az_offset,
                              qr, qi, qj, qk,
                              gy - gy_offset, gp - gp_offset, gr - gr_offset,
                              hpa)

                    sector_row_count += 1
                    i += 1

            # ZERO-FILL unused rows
            if sector_row_count < ROWS_PER_SECTOR:
                start_fill = sector_row_count * BYTES_PER_ROW
                sector_buffer[start_fill:DATA_SIZE] = b"\x00" * (DATA_SIZE - start_fill)

            if lift_trigger_us > 0 and lift_trigger_ms == 0.0:
                lift_trigger_ms = get_synced_ms(lift_trigger_us)

            if top_flame_trigger_us > 0 and top_flame_trigger_ms == 0.0:
                top_flame_trigger_ms = get_synced_ms(top_flame_trigger_us)

            # add new metadata
            struct.pack_into("<IffffI", sector_buffer, CUSTOM_DATA_OFFSET,
                             sector_idx,
                             lift_trigger_ms, top_flame_trigger_ms,
                             max_celsius_during_sector, max_celsius_ts_ms,
                             min_accuracy)

            # Calculate CRC over everything EXCEPT the CRC's last 4 bytes (4092 bytes total)
            crc = crc32(memoryview(sector_buffer)[:4092])
            struct.pack_into("<I", sector_buffer, CRC_OFFSET, crc)

            # Write sector to flash:  bytes 0-4091 are data, last 4 bytes are CRC or 0x00 padding
            f.write(sector_buffer)  # Write exactly 4 KiB

            # Flush every other sector (about 1 sec)
            if sector_idx % 2 == 0:
                f.flush()

            # Debug
            # print(f"\nSector {sector_idx}: stats")
            # print(f"sector_row_count: {sector_row_count} of {ROWS_PER_SECTOR}")
            # print(f"Sector Max Celsius: {max_celsius_during_sector:.2f}° C at {max_celsius_ts_ms} ms")
            # print(f"Sector Min Accuracy (Lin_acc): {min_accuracy}")
            # print(f"CRC: {hex(crc)}")
            #
            # if lift_trigger_us > 0 or top_flame_trigger_us > 0:
            #     us_delta = ticks_diff(top_flame_trigger_us, lift_trigger_us)
            #     print(f"\nLift Time (ms): {lift_trigger_ms} ms")
            #     print(f"Top Flame Time: {top_flame_trigger_ms} ms")
            #     print(f"Lift - Top Frame: {us_delta / 1000.0:.5f} ms")

            sector_idx += 1
            # Debug timing for each write, measured 45 ms
            # write_time = ticks_diff(ticks_ms(), write_start)
            # print(f"Sector flushed (4 KiB). Write: {write_time} ms. Total Rows so far: {i}")

        f.flush()
        os.sync()

    last_sensor_ms = ts_ms
    pico_ms = ticks_diff(ticks_ms(), start)

    print(f"\nFinal flush and sync. Total rows: {i} Rows")
    print(
        f"Sensor timestamps {last_sensor_ms=} {first_sensor_ms=}  sensor duration: {(last_sensor_ms - first_sensor_ms) / 1000:.1f} s")
    print(f"Sensor msec/reports = {(last_sensor_ms - first_sensor_ms) / rows:.2f} ms")

    print(f"Clock msec/reports  = {(pico_ms / rows):.2f} ms")

    kbytes = (BYTES_PER_ROW * rows) / 1024
    print(f"{BYTES_PER_ROW=}, data size = {(rows * BYTES_PER_ROW)} ({kbytes:.1f} KiB)")
    print(f"xfer = {kbytes / (pico_ms / 1000.0):.1f} KiB/s")


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
        if idx == bins:  # Handle maximum value
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

    Ax bias = median(ax) = 0.0000000

    Histogram (Range: -0.042969 to 0.046875)
    ---------------------------------------------------------------------------
     -0.042969 to  -0.036979 |     3 | ###
     -0.036979 to  -0.030990 |     8 | ########
     -0.030990 to  -0.025000 |     3 | ###
     -0.025000 to  -0.019010 |    18 | ##################
     -0.019010 to  -0.013021 |     9 | #########
     -0.013021 to  -0.007031 |    28 | ############################
     -0.007031 to  -0.001042 |    14 | ##############
     -0.001042 to   0.004948 |    39 | ########################################
      0.004948 to   0.010937 |    13 | #############
      0.010937 to   0.016927 |    34 | ##################################
      0.016927 to   0.022917 |    11 | ###########
      0.022917 to   0.028906 |    13 | #############
      0.028906 to   0.034896 |     3 | ###
      0.034896 to   0.040885 |     3 | ###
      0.040885 to   0.046875 |     1 | #

    Ay bias = median(ay) = 0.0000000

    Histogram (Range: -0.054688 to 0.039062)
    ---------------------------------------------------------------------------
     -0.054688 to  -0.048437 |     1 | #
     -0.048437 to  -0.042187 |     0 | 
     -0.042188 to  -0.035938 |     1 | #
     -0.035938 to  -0.029687 |     6 | ######
     -0.029687 to  -0.023438 |     7 | #######
     -0.023438 to  -0.017188 |    26 | ##########################
     -0.017187 to  -0.010937 |    17 | #################
     -0.010938 to  -0.004688 |    17 | #################
     -0.004687 to   0.001563 |    39 | ########################################
      0.001563 to   0.007813 |    17 | #################
      0.007813 to   0.014062 |    29 | #############################
      0.014063 to   0.020313 |    23 | #######################
      0.020313 to   0.026563 |     5 | #####
      0.026563 to   0.032813 |     7 | #######
      0.032812 to   0.039062 |     5 | #####

    Az bias = median(az) = 0.0039063

    Histogram (Range: -0.062500 to 0.062500)
    ---------------------------------------------------------------------------
     -0.062500 to  -0.054167 |     2 | #
     -0.054167 to  -0.045833 |     2 | #
     -0.045833 to  -0.037500 |     3 | #
     -0.037500 to  -0.029167 |     6 | ###
     -0.029167 to  -0.020833 |    14 | #########
     -0.020833 to  -0.012500 |    23 | ##############
     -0.012500 to  -0.004167 |     7 | ####
     -0.004167 to   0.004167 |    62 | ########################################
      0.004167 to   0.012500 |    10 | ######
      0.012500 to   0.020833 |    33 | #####################
      0.020833 to   0.029167 |    15 | #########
      0.029167 to   0.037500 |    11 | #######
      0.037500 to   0.045833 |     6 | ###
      0.045833 to   0.054167 |     2 | #
      0.054167 to   0.062500 |     4 | ##

    ascii Histograms of Gyroscope Biases (rad/s):

    Gy bias = median(gy) = 0.0000000

    Histogram (Range: 0.000000 to 0.100000)
    ---------------------------------------------------------------------------
      0.000000 to   0.006667 |   200 | ########################################
      0.006667 to   0.013333 |     0 | 
      0.013333 to   0.020000 |     0 | 
      0.020000 to   0.026667 |     0 | 
      0.026667 to   0.033333 |     0 | 
      0.033333 to   0.040000 |     0 | 
      0.040000 to   0.046667 |     0 | 
      0.046667 to   0.053333 |     0 | 
      0.053333 to   0.060000 |     0 | 
      0.060000 to   0.066667 |     0 | 
      0.066667 to   0.073333 |     0 | 
      0.073333 to   0.080000 |     0 | 
      0.080000 to   0.086667 |     0 | 
      0.086667 to   0.093333 |     0 | 
      0.093333 to   0.100000 |     0 | 

    Gp bias = median(gp) = 0.0000000

    Histogram (Range: 0.000000 to 0.100000)
    ---------------------------------------------------------------------------
      0.000000 to   0.006667 |   200 | ########################################
      0.006667 to   0.013333 |     0 | 
      0.013333 to   0.020000 |     0 | 
      0.020000 to   0.026667 |     0 | 
      0.026667 to   0.033333 |     0 | 
      0.033333 to   0.040000 |     0 | 
      0.040000 to   0.046667 |     0 | 
      0.046667 to   0.053333 |     0 | 
      0.053333 to   0.060000 |     0 | 
      0.060000 to   0.066667 |     0 | 
      0.066667 to   0.073333 |     0 | 
      0.073333 to   0.080000 |     0 | 
      0.080000 to   0.086667 |     0 | 
      0.086667 to   0.093333 |     0 | 
      0.093333 to   0.100000 |     0 | 

    Gr bias = median(gr) = 0.0000000

    Histogram (Range: 0.000000 to 0.100000)
    ---------------------------------------------------------------------------
      0.000000 to   0.006667 |   200 | ########################################
      0.006667 to   0.013333 |     0 | 
      0.013333 to   0.020000 |     0 | 
      0.020000 to   0.026667 |     0 | 
      0.026667 to   0.033333 |     0 | 
      0.033333 to   0.040000 |     0 | 
      0.040000 to   0.046667 |     0 | 
      0.046667 to   0.053333 |     0 | 
      0.053333 to   0.060000 |     0 | 
      0.060000 to   0.066667 |     0 | 
      0.066667 to   0.073333 |     0 | 
      0.073333 to   0.080000 |     0 | 
      0.080000 to   0.086667 |     0 | 
      0.086667 to   0.093333 |     0 | 
      0.093333 to   0.100000 |     0 | 

    Static Acceleration Biases: AX_BIAS=+0.000000, AY_BIAS=+0.000000, AZ_BIAS=+0.003906
    Static Gyro Biases:         GY_BIAS=+0.000000, GP_BIAS=+0.000000, GR_BIAS=+0.000000

    """
    global AX_BIAS, AY_BIAS, AZ_BIAS, GY_BIAS, GP_BIAS, GR_BIAS

    print(f"Calculating static bias from {samples} samples.")
    print("\n* DO NOT MOVE SENSOR...")

    ax = array('f', [0.0] * samples)
    ay = array('f', [0.0] * samples)
    az = array('f', [0.0] * samples)
    gy = array('f', [0.0] * samples)
    gp = array('f', [0.0] * samples)
    gr = array('f', [0.0] * samples)

    for i in range(20):
        bno.update_sensors()
        if bno.linear_acceleration.updated:
            new_ax, new_ay, new_az = bno.linear_acceleration
            new_gy, new_gp, new_gr = bno.gyro

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
    print(f"\nAx bias = median(ax) = {AX_BIAS:.7f}")
    ascii_histogram(ax, bins=15)
    print(f"\nAy bias = median(ay) = {AY_BIAS:.7f}")
    ascii_histogram(ay, bins=15)
    print(f"\nAz bias = median(az) = {AZ_BIAS:.7f}")
    ascii_histogram(az, bins=15)

    print("\nascii Histograms of Gyroscope Biases (rad/s):")
    print(f"\nGy bias = median(gy) = {GY_BIAS:.7f}")
    ascii_histogram(gy, bins=15)
    print(f"\nGp bias = median(gp) = {GP_BIAS:.7f}")
    ascii_histogram(gp, bins=15)
    print(f"\nGr bias = median(gr) = {GR_BIAS:.7f}")
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

    # Configure pins with Pull-Up resistors so they stay High until grounded
    pin_lift_trig = Pin(22, Pin.IN)
    pin_top_flame = Pin(21, Pin.IN)

    # Attach interrupts for FALLING edge (High to Low)
    pin_lift_trig.irq(trigger=Pin.IRQ_FALLING, handler=handle_lift_interrupt)
    pin_top_flame.irq(trigger=Pin.IRQ_FALLING, handler=handle_top_flame)

    # SPI & I2C
    spi = SPI(0, baudrate=3000000, sck=Pin(18), mosi=Pin(19), miso=Pin(16))
    bno = BNO08X_SPI(spi, cs_pin, reset_pin, int_pin, wake_pin, debug=False)
    i2c = I2C(id=0, scl=Pin(13), sda=Pin(12), freq=400_000)
    bmp = bmpxxx.BMP585(i2c=i2c, address=0x47)

    print(spi)  # baudrate=3000000 required
    print(i2c)  # baudrate= 400000 required
    print("Start")
    print("====================================\n")

    # Set up Barometer
    bmp.pressure_oversample_rate = bmp.OSR4
    bmp.temperature_oversample_rate = bmp.OSR4
    bmp.iir_coefficient = bmp.COEF_1

    # Set known altitude in meters and the sea level pressure (SLP) will be calculated
    bmp.altitude = SITE_ELEVATION
    print(f"Altitude = {bmp.altitude:.2f} meters")
    print(f"Adjusted SLP based on known altitude = {bmp.sea_level_pressure:.2f} hPa\n")

    # Update frequency in Hz, 200Hz = 5ms sample
    # very slow for orientation testing: 10Hz = 100ms
    update_frequency = 200
    bno.linear_acceleration.enable(update_frequency)
    bno.gyro.enable(update_frequency)
    bno.quaternion.enable(update_frequency)

    bno.print_report_period()
    print("BNO08x sensors enabled\n")

    # Calibration :CAUTION: check calibration will NOT time out if inaccurate
    sensor_calibration(bno, stable_sec=3)
    print("\nSleeping for 5 sec - Make sure sensor still for Bias calibration")
    sleep_ms(5000)

    # Sensor Bias calculation: DO Not move sensor, 1 second or 200 samples at 200 Hz
    print("\nStarting Bias calibration...")
    get_static_bias(bno, samples=1 * update_frequency)
    print(f"\nStatic Acceleration Biases: {AX_BIAS=:+.6f}, {AY_BIAS=:+.6f}, {AZ_BIAS=:+.6f}")
    print(f"Static Gyro Biases:         {GY_BIAS=:+.6f}, {GP_BIAS=:+.6f}, {GR_BIAS=:+.6f}")

    # GC after calibration & Bias calculation
    print(f"\nFree memory before gc.collect: {gc.mem_free()} bytes")
    gc.collect()
    print(f"Free memory after  gc.collect: {gc.mem_free()} bytes")

    # Log results

    # 5 ms sample period generates 200 rows/sec (~85 rows) which is 0.5 sec per sector
    duration_seconds = 5
    rows = duration_seconds * update_frequency
    rows = 600_000
    print(f"\nSensor Collection started: {duration_seconds=}, {rows=}, {update_frequency=} Hz,")

    # WRITE-TO-FLASH in 4 KiB sectors, unfortunately 100ms jitter at writes
    write_results_by_sector(bno, bmp, rows, SENSOR_FILE_NAME)

    # write_metadata


if __name__ == "__main__":
    main()
