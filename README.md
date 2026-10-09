# Attitude Estimation via EKF, UKF, ESEKF and EEKF

MATLAB implementation of four Kalman-type filters for estimating attitude (orientation) together with gyroscope and accelerometer biases from IMU (and magnetometer) readings:

- **EKF**: extended Kalman filter (quaternion state)
- **ESEKF**: error-state extended Kalman filter
- **UKF**: unscented Kalman filter
- **EEKF**: Enhanced EKF, which enforces statistical consistency by solving a small semidefinite program whenever the normalized innovation squared exceeds its confidence bound

The filters are tested on synthetic data (Gaussian and Student's-t magnetometer noise) and on the [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/euroc-mav/).

**For an explanation of the method and the results, see the project page:**
**https://mhyoosefian.github.io/projects/attitude-estimation/**

## Requirements

- MATLAB R2020a or newer (for `exportgraphics`)
- Robotics System Toolbox (for `quat2eul` and `wrapToPi`)
- Statistics and Machine Learning Toolbox (for `trnd`, Student's-t scenario only)
- [YALMIP](https://yalmip.github.io/) and [MOSEK](https://www.mosek.com/) (free academic license), **required for the EEKF**

## Repository structure

```
simulationCodes/
  Gaussian/            synthetic data, Gaussian measurement noise
  Student/             synthetic data, Student's-t magnetometer noise
    runMe4filters.m    main script: runs all four filters and makes the plots
    EKF.m, ESEKF.m, UKF.m, EEKF.m   one file per filter
    compute_D.m        terms needed for the posterior Cramér–Rao bound (PCRB)
    utils/             state-transition and measurement functions and their Jacobians
    figs/              results (overwritten when the script runs)
eurocCodes/
  runMe.m              main script for the EuRoC MAV dataset (EKF, UKF, EEKF)
  imu_readings.mat     IMU data from sequence MH_01
  GT.mat               ground truth from sequence MH_01
  img_timeStamps.mat   camera time stamps (used only for the video)
  utils/, figs/        as above
```

## Running the synthetic experiments

1. Open MATLAB in `simulationCodes/Gaussian` (or `simulationCodes/Student`).
2. Run `runMe4filters.m`.

Each script runs 500 Monte Carlo trials of 200 time steps. When it finishes, the figures in `figs/` are regenerated (RMSE of the states and of yaw, pitch and roll, the PCRB comparison for the ESEKF, filter efficiency, and ANIS consistency plots), and the average computation time of each filter is printed in the Command Window.

## Running the EuRoC experiment

1. Open MATLAB in `eurocCodes`.
2. Run `runMe.m`.

The IMU readings and ground truth of sequence MH_01 are already included as `.mat` files, so **no download is needed to run the filters**. The script regenerates the figures in `figs/` (RMSE of yaw, pitch and roll and the ANIS plots) and prints the average computation time of each filter.

### Making the animation video (optional)

The script can also produce a video showing the onboard camera images next to the estimated and true angles. This needs the camera images from the EuRoC dataset:

1. Download sequence **MH_01_easy** (ASL format) from the [EuRoC MAV dataset page](https://projects.asl.ethz.ch/datasets/euroc-mav/).
2. Copy its `cam0` folder into `eurocCodes/`, so that the images are in `eurocCodes/cam0/data/*.png`.
3. Set `make_video = 1` at the top of `runMe.m` and run it. The video is saved as `attitude_animation.mp4`.

## What to expect

- With Gaussian noise, the EEKF and UKF achieve the lowest attitude error, and the EEKF remains consistent.
- With heavy-tailed (Student's-t) magnetometer noise, all filters degrade, and yaw degrades the most.
- On EuRoC, which has no magnetometer, yaw is unobservable and drifts for all filters. Pitch and roll show chattering because the drone's accelerations violate the assumption that body acceleration is negligible compared with gravity; the EEKF stays consistent but is much slower, since it solves a semidefinite program at many time steps.

## Reference

This code was developed as the final project of EECE 7398 Bayesian Filtering at Northeastern University (Fall 2025). The EuRoC data are from:

> M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari, M. W. Achtelik and R. Siegwart, "The EuRoC micro aerial vehicle datasets," *The International Journal of Robotics Research*, 35(10), 2016.
