# rocket data analysis

Analysis of Accelerometer & Gyrometer data pipeline analysis. the analysis pipeline uses Butterworth filter to smooth noise.

Data flow: raw → truncated → filtered → bias-corrected → inertial.

## Use case:

Either use run function in Pycharm or...

Run python to creates plots in ./plots and frames in ~/Downloads:

    $ python3 analysis.py

Run script to create video in ./video from ~/Downloads/frame_0000.png to ~/Downloads/frame_00xx.png 
    
    $ ./video-create.sh

Rocket IMU Data issues Issues:
- there is massive clipping of accelerometer at launch
- low ~250ms sampling rate
- Gyro shows lots of tumbling after chute deploy gimbal lock issues, should use quaternions.
- Gyro data makes post-apogee analysis tricky

## Analysis code based on SOURCE:
THANKS!

https://github.com/cmontalvo251/aerospace/blob/main/rockets/PLAR/post_launch_analysis.py
https://www.youtube.com/watch?v=mb1RNYKtWQE
