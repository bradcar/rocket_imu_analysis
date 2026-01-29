# read_prepare_9_dof.py
# TODO DEBUG: Dual-core didn't solve >100ms jitter during flash writes,  likely better to use serial code
# TODO ADD hPa barometer processing

"""
Linear_acceleration, Quaternion, Gyro logging from BNO086 sensor to flash memory.

This code checks the calibration accuracy of the bno.linear_acceleration, bno.quaternion, and bno.gyro and
continues this until all have a better accuracy >=2 (medium to high). Then saves the calibration to the
BNO086.

After calibration, the bno.linear_acceleration and bno.gyro are measured while not moving to create
a bias correction. The values collected are shown in an ascii histogram. Then the median is used
as the bias correction that is applied to dall data.

Core 0 is reading the sensors and populating a buffer, it signals Core 1 to write sector-sized buffer
Core 1 takes the data in the buffer and writes a sector.

Buffer 4 KiB Sector, then sector to flash. This will show jitter at sector writes.
    write_results_by_sector(bno, rows, filename)
    The max size of storage is limited only by the free space on flash. It gathers sensor result <- 4KiB.
    The 4 KiB limit is due to the Flash's Sector size of 4 KiB. Writing this size is the most efficient.
    With 5ms sample rate, the flash write and intermittant flush (duty cycle settable in code) will cause
    50 ms to 110 ms jitter in sample collection.

The BNO086 sensor is connected to Raspberry Pi Pico 2 W by SPI.

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

import _thread
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

# Global shared queue and state
data_queue = []
stop_logging = False
queue_lock = _thread.allocate_lock()

# Software Trigger: Core 0 waits on this lock
write_trigger = _thread.allocate_lock()

def file_writer_consumer_core1(filename, total_rows):
    global data_queue, stop_logging
    print(f"[Core 1] Consumer active. Target: {total_rows} rows")
    rows_processed = 0
    sector_buffer = bytearray(SECTOR_SIZE)

    try:
        print(f"[Core 1] Opening {filename} for binary write...")
        with open(filename, "wb") as f:
            while True:
                # Wait for Core 0 to signal that data is ready
                write_trigger.acquire() 
                
                # Process all full sectors available in the queue
                while len(data_queue) >= ROWS_PER_SECTOR:
                    with queue_lock:
                        batch = data_queue[:ROWS_PER_SECTOR]
                        del data_queue[:ROWS_PER_SECTOR]

                    # Clear buffer and pack data
                    sector_buffer[:] = b"\x00" * SECTOR_SIZE
                    for row_idx, row_data in enumerate(batch):
                        struct.pack_into(pack_string, sector_buffer, row_idx * BYTES_PER_ROW, *row_data)

                    # Calculate CRC32 for the data portion (DATA_SIZE = 4092)
                    crc = binascii.crc32(memoryview(sector_buffer)[:DATA_SIZE])
                    # Store CRC in the last 4 bytes of the 4096 sector
                    struct.pack_into("<I", sector_buffer, DATA_SIZE, crc)
                    
                    # Write to Flash
                    f.write(sector_buffer)
                    # f.flush() 
                    
                    rows_processed += ROWS_PER_SECTOR
                    if rows_processed % (ROWS_PER_SECTOR * 5) == 0:
                        print(f"[Core 1 Progress] Written: {rows_processed}/{total_rows}")

                # --- EXIT & DRAIN CONDITION ---
                if stop_logging:
                    # Final check for any remaining partial sector
                    if len(data_queue) > 0:
                        print(f"[Core 1 Drain] Processing final {len(data_queue)} rows...")
                        with queue_lock:
                            final_batch = data_queue[:]
                            del data_queue[:]

                        sector_buffer[:] = b"\x00" * SECTOR_SIZE # Pad remainder with zeros
                        for row_idx, row_data in enumerate(final_batch):
                            struct.pack_into(pack_string, sector_buffer, row_idx * BYTES_PER_ROW, *row_data)

                        # Final CRC for the partial block
                        crc = binascii.crc32(memoryview(sector_buffer)[:DATA_SIZE])
                        struct.pack_into("<I", sector_buffer, DATA_SIZE, crc)
                        
                        f.write(sector_buffer)
                        f.flush()
                        rows_processed += len(final_batch)
                    
                    break

        # Final filesystem sync
        os.sync()
        print(f"[Core 1] Finished. Total rows saved: {rows_processed}")

    except Exception as e:
        print(f"[Core 1] CRITICAL ERROR: {e}")
    finally:
        print("[Core 1] Worker thread exiting clean.")


def sensor_reader_producer_core0(bno, rows):
    global data_queue, stop_logging
    
    # Check report periods before starting
    print("[Core 0] Starting sensor_reader_producer_core1")

    ax_off, ay_off, az_off = AX_BIAS, AY_BIAS, AZ_BIAS
    gy_off, gp_off, gr_off = GY_BIAS, GP_BIAS, GR_BIAS
    
    update = bno.update_sensors
    lin_acc = bno.linear_acceleration
    quat = bno.quaternion
    gyro = bno.gyro

    count = 0
    total_polls = 0
    reports_seen_total = 0
    
    print(f"[Core 1] Entering main loop for {rows} rows")
    
    try:
        while count < rows:
            total_polls += 1
            num_reports = update()

            if num_reports > 0:
                reports_seen_total += num_reports
                
                if lin_acc.updated:
                    ax, ay, az, _, ts_ms = lin_acc.full
                    qr, qi, qj, qk = quat 
                    gy, gp, gr = gyro

                    with queue_lock:
                        data_queue.append((
                            ts_ms,
                            ax - ax_off, ay - ay_off, az - az_off,
                            qr, qi, qj, qk,
                            gy - gy_off, gp - gp_off, gr - gr_off
                        ))
                    
                    count += 1
                    
                    # Signal Core 0 when a sector is ready
                    if len(data_queue) >= ROWS_PER_SECTOR:
                        if write_trigger.locked():
                            # print(f"[Core 1] Sector Ready {len(data_queue)=}. Releasing Core 0.")
                            write_trigger.release()

            if total_polls % 2000 == 0 and reports_seen_total == 0:
                print(f"[Core 0 Silence] {total_polls} polls performed. 0 reports received.")
            
            if num_reports == 0:
                sleep_us(100)

    except Exception as e:
        print(f"\n[Core 1] !!! CRITICAL EXCEPTION !!!: {e}\n")
    finally:
        stop_logging = True
        if write_trigger.locked():
            write_trigger.release()
        print(f"[Core 0] Finished. Polls: {total_polls}, Reports: {reports_seen_total}, Rows: {count}")


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

    bias will use median(ax) = 0.0000000

    Histogram (Range: -0.046875 to 0.054688)
    ---------------------------------------------------------------------------
     -0.046875 to  -0.040104 |     2 | #
     -0.040104 to  -0.033333 |     1 | 
     -0.033333 to  -0.026562 |     6 | #####
     -0.026563 to  -0.019792 |     7 | #####
     -0.019792 to  -0.013021 |    26 | #####################
     -0.013021 to  -0.006250 |    21 | #################
     -0.006250 to   0.000521 |    48 | ########################################
      0.000521 to   0.007292 |    20 | ################
      0.007292 to   0.014063 |    31 | #########################
      0.014063 to   0.020833 |    16 | #############
      0.020833 to   0.027604 |    11 | #########
      0.027604 to   0.034375 |     1 | 
      0.034375 to   0.041146 |     7 | #####
      0.041146 to   0.047917 |     2 | #
      0.047917 to   0.054688 |     1 | 

    bias will use median(ay) = 0.0019531

    Histogram (Range: -0.031250 to 0.046875)
    ---------------------------------------------------------------------------
     -0.031250 to  -0.026042 |     7 | ######
     -0.026042 to  -0.020833 |     6 | #####
     -0.020833 to  -0.015625 |     5 | ####
     -0.015625 to  -0.010417 |    22 | #####################
     -0.010417 to  -0.005208 |     9 | ########
     -0.005208 to   0.000000 |    24 | #######################
      0.000000 to   0.005208 |    41 | ########################################
      0.005208 to   0.010417 |    19 | ##################
      0.010417 to   0.015625 |    15 | ##############
      0.015625 to   0.020833 |    28 | ###########################
      0.020833 to   0.026042 |    10 | #########
      0.026042 to   0.031250 |     4 | ###
      0.031250 to   0.036458 |     7 | ######
      0.036458 to   0.041667 |     2 | #
      0.041667 to   0.046875 |     1 | 

    bias will use median(az) = -0.0039063

    Histogram (Range: -0.058594 to 0.046875)
    ---------------------------------------------------------------------------
     -0.058594 to  -0.051562 |     3 | ##
     -0.051562 to  -0.044531 |     1 | 
     -0.044531 to  -0.037500 |     7 | ######
     -0.037500 to  -0.030469 |    16 | ###############
     -0.030469 to  -0.023438 |     0 | 
     -0.023438 to  -0.016406 |    18 | #################
     -0.016406 to  -0.009375 |    30 | ############################
     -0.009375 to  -0.002344 |    42 | ########################################
     -0.002344 to   0.004687 |     6 | #####
      0.004687 to   0.011719 |    29 | ###########################
      0.011719 to   0.018750 |    26 | ########################
      0.018750 to   0.025781 |     4 | ###
      0.025781 to   0.032812 |     9 | ########
      0.032812 to   0.039844 |     5 | ####
      0.039844 to   0.046875 |     4 | ###

    ascii Histograms of Gyroscope Biases (rad/s):

    bias will use median(gy) = 0.0000000

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

    bias will use median(gp) = 0.0000000

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

    bias will use median(gr) = 0.0000000

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

    Static Acceleration Biases: AX_BIAS=+0.000000, AY_BIAS=+0.001953, AZ_BIAS=-0.003906
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
    # very slow for orientation testing: 10Hz = 100ms
    update_frequency = 200
    bno.linear_acceleration.enable(update_frequency)
    bno.gyro.enable(update_frequency)
    bno.quaternion.enable(update_frequency)

    bno.print_report_period()
    print("BNO08x sensors enabled\n")

    # Check Calibration
    # CAUTION: this will NOT time out if inaccurate
    sensor_calibration(bno, stable_sec=3)

    # Calculate Sensor Bias DO NOR move sensor
    print("\nSleeping for 4 sec - Make sure sensor still for Bias calibration")
    sleep_ms(4000)
    print("\nStarting Bias calibration...")
    get_static_bias(bno, samples=1 * update_frequency)
    print(f"\nStatic Acceleration Biases: {AX_BIAS=:+.6f}, {AY_BIAS=:+.6f}, {AZ_BIAS=:+.6f}")
    print(f"Static Gyro Biases:         {GY_BIAS=:+.6f}, {GP_BIAS=:+.6f}, {GR_BIAS=:+.6f}")

    print(f"\nFree Memory before gc.collect: {gc.mem_free()} bytes")
    gc.collect()
    print(f"Free memory before threading:  {gc.mem_free()} bytes")

# --- 4. Parallel Handoff ---
    duration_seconds = 5
    rows_to_collect = duration_seconds * update_frequency
    filename = "flight_log_dual_core.bin"

    print(f"PREPARING THREADS: Collecting {rows_to_collect} rows.")

    duration_seconds = 5
    rows_to_collect = duration_seconds * update_frequency
    filename = "flight_log_dual_core.bin"

    print(f"Start cores: Core 0 (Sensor reader), Core 1 (Flash writer)")

    # Ensure the trigger is locked so Core 1 waits
    if not write_trigger.locked():
        write_trigger.acquire()

    # Launch Core 1 as the Consumer Sensor-reader loop
    print("\n[System] Launch Core 1: file_writer_consumer_core1")
    _thread.start_new_thread(file_writer_consumer_core1, (filename, rows_to_collect))

    # Launch Core 0 as the Producer flash-writer loop
    print("[System] Launch Core 0 : sensor_reader_producer_core0")
    sensor_reader_producer_core0(bno, rows_to_collect)

    sleep_ms(200)
    print("\n[System] All tasks complete.")

if __name__ == "__main__":
    main()
