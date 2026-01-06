# rocket data analysis

Analysis of Accelerometer & Gyrometer data pipeline analysis. the analysis pipeline uses Butterworth filter to smooth noise.

Data flow: raw → truncated → filtered → bias-corrected → inertial.

Issues:
- there is massive clipping of accelerometer at launch
- low ~250ms sampling rate
- gyro shows lots of tumbling after chute deploy gimbal lock issues, should use quaternions.
- gyro data makes post apogee analysis tricky

based on SOURCE:
# https://github.com/cmontalvo251/aerospace/blob/main/rockets/PLAR/post_launch_analysis.py
