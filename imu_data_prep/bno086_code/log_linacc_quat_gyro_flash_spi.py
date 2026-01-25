# Linear_acceleration, Quaternion, Gyro logging to flash BNO08x MicroPython SPI
#
# CAUTION: TIME IN msec NOT SECONDS, for BNO efficiency
# CAUTION: CSV IS CONVERTED TO SECONDS
#
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
#

import binascii  # For fast CRC32
import gc
import os
import struct

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


def write_results_whole_batch(bno, rows: int, filename: str):
    """
    Buffer all results in memory then write whole file to flash. The advantage is very little jitter
    for high-frequency updates (5 ms) with little jitter.
    
    Low jitter, but limited to only 95 KiB in-memory storage.
    This function usefor for simple tests, but for practical use use sector-format version.

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
            f"need {required_buffer_size/1024.0:.1f} KiB, have {(free_heap_size - SAFETY_MARGIN)/1024.0:.1f} KiB free. "
            f"Max rows={(free_heap_size - SAFETY_MARGIN) / BYTES_PER_ROW:.0f} "
            "Use sector-based logging instead."
        )
    else:
        percent_memory =(required_buffer_size/free_heap_size)*100.0
        print(f"Memory used for buffer: {percent_memory:.0f}%")

    buffer = bytearray(required_buffer_size)

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
            pack_into(pack_string, buffer, offset, ts_ms, ax, ay, az, qr, qi, qj, qk, gy, gp, gr)

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
    # Time Packing data into a 4 KiB buffer & writing sectors for flash
    print("\nWriting data to Flash in 4 KiB sector chunks to flash")

    # Reset file in sector data format
    with open(filename, "wb") as f:
        pass

    # Buffer of exactly 4 KiB, data: 4092 CRC: last 4 bytes
    sector_buffer = bytearray(SECTOR_SIZE)

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
                              ts_ms, ax, ay, az, qr, qi, qj, qk, gy, gp, gr)

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

        if all(x >= 2 for x in (accel_accuracy,  gyro_accuracy, quat_accuracy)):
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

    # Update frequency in Hz, 200Hz = 5ms sample, 10Hz = 100ms
    update_frequency = 10
    bno.linear_acceleration.enable(update_frequency)
    bno.gyro.enable(update_frequency)
    bno.quaternion.enable(update_frequency)

    bno.print_report_period()
    print("BNO08x sensors enabled\n")

    # --- Check Calibration
    # CAUTION this will NOT time out if inaccurate
    sensor_calibration(bno, stable_sec=3)

    # --- Log results - TWO Options - typically chose one

    # 5ms sample period generate 200 rows/sec
    duration_seconds = 20
    rows = duration_seconds * update_frequency
    print(f"{duration_seconds=}, {rows=}, {update_frequency=}Hz,")

    # Buffer all data before writes - limited to 95 KiB logs
    filename = "flight_log_2026xxxx_xpm_whole.bin"
    write_results_whole_batch(bno, rows, filename)

    # Buffer Sectors, and write by sector. This will show jitter at sector writes.
    filename = "flight_log_2026xxxx_xpm_sector.bin"
    write_results_by_sector(bno, rows, filename)


if __name__ == "__main__":
    main()
