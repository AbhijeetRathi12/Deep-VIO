% Used to introduce noise in actual IMU readings (obtained from Blender by simulating) to make IMU readings more practical. 
clear all
close all
load states.mat
xyz = state(:, 1:3);
vxyz = state(:, 4:6);
quat = state(:, 7:10);
angVel = state(:, 11:13);
time = time(1,:);

% Calculate acceleration
delta_time = 0.01;
delta_vxyz = diff(vxyz);
accl = delta_vxyz / delta_time;

fs = 1000;
totalNumSamples = 5000;
IMU = imuSensor('accel-gyro','SampleRate', fs);
angVel = angVel(1:end-2, :);
accl = accl(1:end-1, :);
[accelReadings,gyroReadings] = IMU(accl,angVel);

% Concatenate accelReadings and gyroReadings into a single matrix
sensorData = [accelReadings, gyroReadings];

% Save the combined sensor data to a CSV file
csvFileName = 'IMU_data_file.csv';
writematrix(sensorData, csvFileName);

xyz = xyz(1:end-2, :);
quat = quat(1:end-2, :);
eul = quat2eul(quat);
PoseData = [xyz, eul];
% Save the combined sensor data to a CSV file
csvFileName = 'Pose_data.csv';
writematrix(PoseData, csvFileName);