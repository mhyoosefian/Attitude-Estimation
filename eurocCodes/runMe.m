clc
clear
close
make_video = 0;

%% Loading EuRoC data

% IMU data
IMUReadings = load('imu_readings.mat');
imu_data = IMUReadings.data;
time_stamp_imu = imu_data(:, 1);

% Ground-truth data
GT = load('GT.mat');                           
grt_data = GT.data;
time_stamp_grt = grt_data(:, 1);

% Image data (just for video making)
time_stamp_img = load('img_timeStamps.mat');
time_stamp_img = time_stamp_img.data;

% Find the common timestamp
[common_ts, idx1, idx2] = intersect(time_stamp_grt, time_stamp_img, 'stable');

image_idx = 500;
data_start = 10*(image_idx - 23);
time_stamp_img = time_stamp_img(image_idx:end);

imu_data = imu_data(222 + data_start:end, 2:end);
time_stamp_imu = time_stamp_imu(222 + data_start:end);
gyroX = imu_data(:,1); 
gyroY = imu_data(:,2);
gyroZ = imu_data(:,3);
accelX = imu_data(:,4);
accelY = imu_data(:,5);
accelZ = imu_data(:,6);


grt_data = grt_data(6 + data_start:end, 2:end);
time_stamp_grt = time_stamp_grt(6 + data_start:end);
grt_quat = grt_data(:,4:7);
grt_bg = grt_data(:,11:13);
grt_ba = grt_data(:,14:16);

N = size(grt_data, 1) - 1;                        % time steps

%% Initialization

addpath utils/
inpuDim = 5;
outputDim = 3;
dt = 1/200;                     % sampling time
g_ref = -9.81;

% Noise statistics
processNoiseDim = 9;
sig_g = 0.005;                   % gyro noise std
sig_b_g = 0.001;                % gyro bias noise std
sig_a = 0.07;                   % accel noise std
sig_b_a = 0.01;                % accel bias noise std


Q = blkdiag(sig_g^2*eye(3) , sig_b_g^2*eye(3) , sig_b_a^2*eye(3));
Q = Q*dt^2;
sqrtQ = sqrtm(Q);
R = sig_a^2*eye(3);
sqrtR = sqrtm(R);
P0 = 1e-2*eye(processNoiseDim + 1);

% EKF intialization
P_ekf = cell(N+1, 1);
P_ekf{1} = P0;

% EEKF intialization
P_eekf = cell(N+1, 1);
P_eekf{1} = P0;
eps_avg = 0;

% UKF initialization
stateDim = 10;
P_ukf = cell(N+1, 1);
P_ukf{1} = P0;
d = 9;
L = d + outputDim + stateDim;
alpha = 1;                      % primary scaling parameter
beta = 2;                       % secondary scaling parameter
kappa = 0;                      % teritary scaling parameter

%% Main code

xhat_ekf = cell(N+1, 1);
xhat_ukf = cell(N+1, 1);
xhat_eekf = cell(N+1, 1);
eps2_ekf = zeros(N, 1);
eps2_ukf = zeros(N, 1);
eps2_eekf = zeros(N, 1);
runtime_ekf = zeros(N, 1);
runtime_ukf = zeros(N, 1);
runtime_eekf = zeros(N, 1);
xtrue = cell(N, 1);

% Initial condition
x(:, 1) = [grt_quat(1, :) grt_bg(1, :) grt_ba(1, :)]';
xtrue{1} = x(:, 1);
mu_ekf(:, 1) = x(:, 1); % + sqrtm(P0)*randn(stateDim, 1);
mu_ukf(:, 1) = x(:, 1); % + sqrtm(P0)*randn(stateDim, 1);
mu_eekf(:, 1) = x(:, 1); % + sqrtm(P0)*randn(stateDim, 1);
xhat_ekf{1} = mu_ekf(:, 1);
xhat_ukf{1} = mu_ukf(:, 1);
xhat_eekf{1} = mu_eekf(:, 1);

% Simulate the estimators
for t = 1:N
    
    if rem(t, 100) == 0
        disp([num2str(100*t/N) , '% done!']);
    end
    % Simulate the system
    u = [gyroX(t); gyroY(t); gyroZ(t)];
    u = [u; g_ref; dt];
    x(:, t+1) = [grt_quat(t+1, :) grt_bg(t+1, :) grt_ba(t+1, :)]';
    xtrue{t+1} = x(:, t+1);
    q_true  = x(1:4,  t+1);
    bg_true = x(5:7,  t+1);
    ba_true = x(8:10, t+1);

    % Recieve a measurement
    y(:, t+1) = [accelX(t+1); accelY(t+1); accelZ(t+1)];
%     v = sqrtR*randn(outputDim, 1);
%     y(:, t+1) = h(x(:, t+1), u, v);
    
    % EKF
    tic;
    [mu_ekf(:, t+1), P_ekf{t+1}, eps2_ekf(t)] = EKF(mu_ekf(:, t), P_ekf{t}, Q, R, u, y(:, t+1));
    runtime_ekf(t) = toc;
    xhat_ekf{t+1} = mu_ekf(:, t+1);

    % EEKF
    tic;
    [mu_eekf(:, t+1), P_eekf{t+1}, eps2_eekf(t), ~, eps_avg] = EEKF(mu_eekf(:, t), P_eekf{t}, Q, R, u, y(:, t+1), t, eps_avg);
    runtime_eekf(t) = toc;
    xhat_eekf{t+1} = mu_eekf(:, t+1);

    % UKF
    tic;
    [mu_ukf(:, t+1), P_ukf{t+1}, eps2_ukf(t)] = UKF(mu_ukf(:, t), P_ukf{t}, Q, R, u, y(:, t+1), alpha, beta, kappa);
    runtime_ukf(t) = toc;
    xhat_ukf{t+1} = mu_ukf(:, t+1);
   
end


% Compute average epsilon (NIS)
eps_avg_ekf = cumsum(eps2_ekf) ./ (1:N)';
eps_avg_ukf = cumsum(eps2_ukf) ./ (1:N)';
eps_avg_eekf = cumsum(eps2_eekf) ./ (1:N)';
mean_eps_avg_ekf = eps_avg_ekf;
mean_eps_avg_ukf = eps_avg_ukf;
mean_eps_avg_eekf = eps_avg_eekf;
mean_eps_avg_true = outputDim*ones(N,1);

% Compute runtime
cpu_time_ekf = geo_mean(mean(runtime_ekf, 2));
cpu_time_ukf = geo_mean(mean(runtime_ukf, 2));
cpu_time_eekf = geo_mean(mean(runtime_eekf, 2));

% Compute angle MSE (for both UKF and ES-EKF)
MSE_angles_ekf = cell(1, N);
MSE_angles_ukf = cell(1, N);
MSE_angles_eekf = cell(1, N);
eul_ekf = cell(1, N);
eul_ukf = cell(1, N);
eul_eekf = cell(1, N);
eul_true = cell(1, N);
ang_err_ekf = cell(1, N);
ang_err_ukf = cell(1, N);
ang_err_eekf = cell(1, N);
for k = 1:N+1

    % Quaternions -> Euler (yaw, pitch, roll) in radians
    eul_ekf{k}  = quat2eul(xhat_ekf{k}(1:4)');   % 1x3
    eul_ukf{k}  = quat2eul(xhat_ukf{k}(1:4)');
    eul_eekf{k}  = quat2eul(xhat_eekf{k}(1:4)');
    eul_true{k} = quat2eul(xtrue{k}(1:4)');

    % Wrap angle errors to [-pi, pi]
    ang_err_ekf{k} = wrapToPi(eul_ekf{k}  - eul_true{k});   % 1x3
    ang_err_ukf{k} = wrapToPi(eul_ukf{k}  - eul_true{k});   % 1x3
    ang_err_eekf{k} = wrapToPi(eul_eekf{k}  - eul_true{k});

    MSE_angles_ekf{k} = ang_err_ekf{k}(:) * ang_err_ekf{k}(:).';
    MSE_angles_ukf{k} = ang_err_ukf{k}(:) * ang_err_ukf{k}(:).';
    MSE_angles_eekf{k} = ang_err_eekf{k}(:) * ang_err_eekf{k}(:).';
end
eul_true = cell2mat(eul_true(:));
eul_ekf = cell2mat(eul_ekf(:));
eul_ukf = cell2mat(eul_ukf(:));
eul_eekf = cell2mat(eul_eekf(:));
ang_err_ekf = cell2mat(ang_err_ekf(:));
ang_err_ukf = cell2mat(ang_err_ukf(:));
ang_err_eekf = cell2mat(ang_err_eekf(:));

eul_ekf_vis = eul_true + ang_err_ekf;
eul_ukf_vis = eul_true + ang_err_ukf;
eul_eekf_vis = eul_true + ang_err_eekf;

eul_true_deg = rad2deg(eul_true);
eul_ekf_deg  = rad2deg(eul_ekf_vis);
eul_eekf_deg  = rad2deg(eul_eekf_vis);

%% Plot the results

folderName = 'figs';
if ~exist(folderName, 'dir')
       mkdir(folderName)
end

% Plot average epsilon for EKF
f = figure;
f.Units = "points";
f.Position = [1 1 280 140];
hold on;
box on;
grid on;
set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex');
set(0, 'DefaultColorbarTickLabelInterpreter', 'latex');
set(0, 'DefaultAxesFontName', 'TimesNewRoman');
set(0, 'DefaultTextFontName', 'TimesNewRoman');
set(0, 'DefaultAxesFontSize', 8);
set(0, 'DefaultTextFontSize', 10);
set(0, 'DefaultLineLineWidth', 1.2);
set(0, 'DefaultLineLineStyle', '-');
plot(1:N, mean_eps_avg_ekf, 'color', [0.6350, 0.0780, 0.1840]);
plot(1:N, mean_eps_avg_true, 'color', 'k');
ylabel('$\bar \epsilon^2_k$')
xlabel('Time step')
title('EKF');
% ylim([-2 12]);
axis tight
fileName = 'fig_eps2EKF.pdf';
filePath = fullfile(folderName, fileName);
exportgraphics(f, filePath, 'ContentType', 'vector');

% Plot average epsilon for UKF
z = 1.96;
faceAlpha = 0.3;
f = figure;
f.Units = "points";
f.Position = [1 1 280 140];
hold on;
box on;
grid on;
set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex');
set(0, 'DefaultColorbarTickLabelInterpreter', 'latex');
set(0, 'DefaultAxesFontName', 'TimesNewRoman');
set(0, 'DefaultTextFontName', 'TimesNewRoman');
set(0, 'DefaultAxesFontSize', 8);
set(0, 'DefaultTextFontSize', 10);
set(0, 'DefaultLineLineWidth', 1.2);
set(0, 'DefaultLineLineStyle', '-');
plot(1:N, mean_eps_avg_ukf, 'color', [0, 0.4470, 0.7410]);
plot(1:N, mean_eps_avg_true, 'color', 'k');
ylabel('$\bar \epsilon^2_k$')
xlabel('Time step')
title('UKF');
% ylim([-2 12]);
axis tight
fileName = 'fig_eps2UKF.pdf';
filePath = fullfile(folderName, fileName);
exportgraphics(f, filePath, 'ContentType', 'vector');

% Plot average epsilon for EEKF
f = figure;
f.Units = "points";
f.Position = [1 1 280 140];
hold on;
box on;
grid on;
set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex');
set(0, 'DefaultColorbarTickLabelInterpreter', 'latex');
set(0, 'DefaultAxesFontName', 'TimesNewRoman');
set(0, 'DefaultTextFontName', 'TimesNewRoman');
set(0, 'DefaultAxesFontSize', 8);
set(0, 'DefaultTextFontSize', 10);
set(0, 'DefaultLineLineWidth', 1.2);
set(0, 'DefaultLineLineStyle', '-');
plot(1:N, mean_eps_avg_eekf, 'color', [0.8500, 0.3250, 0.0980]);
plot(1:N, mean_eps_avg_true, 'color', 'k');
ylabel('$\bar \epsilon^2_k$')
xlabel('Time step')
title('EEKF');
% ylim([-2 12]);
axis tight
fileName = 'fig_eps2EEKF.pdf';
filePath = fullfile(folderName, fileName);
exportgraphics(f, filePath, 'ContentType', 'vector');
 
% Show cpu time
disp(['average CPU time for EKF is ' , num2str(cpu_time_ekf), ' (s)']);
disp(['average CPU time for UKF is ' , num2str(cpu_time_ukf), ' (s)']);
disp(['average CPU time for EEKF is ' , num2str(cpu_time_eekf), ' (s)']);

% Plot angle RMSE
for i=1:3
    tmp = cat(3, MSE_angles_ekf{:});
    rmse_angles_ekf = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
    tmp = cat(3, MSE_angles_ukf{:});
    rmse_angles_ukf = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
    tmp = cat(3, MSE_angles_eekf{:});
    rmse_angles_eekf = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
    f = figure;
    f.Units = "points";
    f.Position = [1 1 280 140];
    hold on;
    box on;
    grid on;
    set(0, 'DefaultTextInterpreter', 'latex');
    set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(0, 'DefaultLegendInterpreter', 'latex');
    set(0, 'DefaultColorbarTickLabelInterpreter', 'latex');
    set(0, 'DefaultAxesFontName', 'TimesNewRoman');
    set(0, 'DefaultTextFontName', 'TimesNewRoman');
    set(0, 'DefaultAxesFontSize', 8);
    set(0, 'DefaultTextFontSize', 8);
    set(0, 'DefaultLineLineWidth', 1.2);
    set(0, 'DefaultLineLineStyle', '-');
    plot(0:N, (180/pi).*rmse_angles_ekf, 'Color', [0.6350, 0.0780, 0.1840]);
    plot(0:N, (180/pi).*rmse_angles_ukf, 'Color', [0, 0.4470, 0.7410]);
    plot(0:N, (180/pi).*rmse_angles_eekf, 'Color', [0.8500, 0.3250, 0.0980]);
    legend('EKF', 'UKF', 'EEKF', 'Interpreter', 'latex');
    if i == 1
        ylabel('RMSE of $\psi (^\circ)$')
    elseif i == 2
        ylabel('RMSE of $\theta (^\circ)$')
    else
        ylabel('RMSE of $\phi (^\circ)$')
    end
    xlabel('Time step')
    fileName = sprintf('fig_RMSE_angles_%d.pdf', i);
    filePath = fullfile(folderName, fileName);
    exportgraphics(f, filePath, 'ContentType', 'vector');
end


%% Save results

% folderName = 'results';
% if ~exist(folderName, 'dir')
%        mkdir(folderName)
% end
% filePath = fullfile(folderName, 'workspace.mat');
% save(filePath);

%% Make a video

if make_video == 1
      % ================== ANIMATION SCRIPT ================== %
    % Assumes variables in workspace:
    %   eul_true : N x 3 (yaw, pitch, roll) in radians
    %   eul_ekf  : N x 3 (yaw, pitch, roll) in radians
    %   camFolder: folder with images, e.g., 'cam0'
    %   img_idx  : starting image index (1-based) in sorted list
    %   stride   : number of eul samples per image (e.g., 10)
    
    clc;
    
    % ---------- User parameters ----------
    if ~exist('camFolder','var'); camFolder = 'cam0/data'; end
    if ~exist('stride','var');    stride    = 10;    end
    if ~exist('image_idx','var');   image_idx   = 1;     end
    
    image_ext  = '*.png';           % change to '*.jpg' if needed
    video_name = 'attitude_animation.mp4';
    fps        = 20;                % playback fps
    
    % ---------- Attitude data ----------
    N = size(eul_true, 1);
    if size(eul_ekf,1) ~= N
        error('eul_true and eul_ekf must have the same number of rows');
    end
    
    % time axis (sample index as "time"); use real time if you have it
    t = (0:N-1)';   % column vector
    
    % ---------- Image list ----------
    img_files = dir(fullfile(camFolder, image_ext));
    if isempty(img_files)
        error('No images with extension %s found in folder %s', image_ext, camFolder);
    end
    
    % sort images by name (EuRoC filenames are timestamps)
    [~, idx_sort] = sort({img_files.name});
    img_files = img_files(idx_sort);
    
    num_images_total = numel(img_files);
    
    if image_idx < 1 || image_idx > num_images_total
        error('img_idx (%d) out of range. There are only %d images.', image_idx, num_images_total);
    end
    
    % usable images starting from img_idx
    num_images_from_idx = num_images_total - image_idx + 1;
    
    % how many frames can we support from the eul data?
    % eul index used for frame k: idx_eul = 1 + (k-1)*stride
    max_frames_from_eul = floor((N-1)/stride) + 1;
    
    % final number of frames
    num_frames = min(num_images_from_idx, max_frames_from_eul);
    
    fprintf('Using %d frames (images from %d to %d, eul indices up to %d)\n', ...
        num_frames, image_idx, image_idx + num_frames - 1, 1 + (num_frames-1)*stride);
    
    % ---------- Video writer ----------
    v = VideoWriter(video_name, 'MPEG-4');
    v.FrameRate = fps;
    open(v);
    
    % ---------- Figure and layout ----------
    fig = figure('Units','normalized','Position',[0.05 0.05 0.9 0.8]);
    set(0, 'DefaultTextInterpreter', 'latex');
    set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(0, 'DefaultLegendInterpreter', 'latex');
    set(0, 'DefaultColorbarTickLabelInterpreter', 'latex');
    set(0, 'DefaultAxesFontName', 'TimesNewRoman');
    set(0, 'DefaultTextFontName', 'TimesNewRoman');
    set(0, 'DefaultAxesFontSize', 14);
    set(0, 'DefaultTextFontSize', 14);
    set(0, 'DefaultLineLineWidth', 1.5);
    set(0, 'DefaultLineLineStyle', '-');
    
    % 3 rows, 2 columns:
    %  - left column: image (one axes spanning 3 rows)
    %  - right column: three axes for yaw, pitch, roll
    tl = tiledlayout(fig, 3, 2, 'TileSpacing','compact', 'Padding','compact');
    
    % Left: image spanning all 3 rows in column 1
    ax_img = nexttile(tl, [3 1]);   % span 3 rows, 1 column
    imshow(zeros(480,640,3,'uint8'));
    title(ax_img, 'cam0 view');
    
    % Right column: 3 stacked axes (one per row) in column 2
    ax1 = nexttile(tl);   % row 1, col 2
    ax2 = nexttile(tl);   % row 2, col 2
    ax3 = nexttile(tl);   % row 3, col 2
    
    hold(ax1,'on'); grid(ax1,'on');
    hold(ax2,'on'); grid(ax2,'on');
    hold(ax3,'on'); grid(ax3,'on');
    
    % Pre-create line objects
    h_true_yaw   = plot(ax1, NaN, NaN, 'b-');
    h_ekf_yaw    = plot(ax1, NaN, NaN, 'r-');
    h_true_pitch = plot(ax2, NaN, NaN, 'b-');
    h_ekf_pitch  = plot(ax2, NaN, NaN, 'r-');
    h_true_roll  = plot(ax3, NaN, NaN, 'b-');
    h_ekf_roll   = plot(ax3, NaN, NaN, 'r-');
    
    ylabel(ax1, '$\psi (^\circ)$');
    ylabel(ax2, '$\theta (^\circ)$');
    ylabel(ax3, '$\phi (^\circ)$');
    xlabel(ax3, 'time step');
    
    legend(ax1, 'True', 'EKF', 'Location','best');
    legend(ax2, 'True', 'EKF', 'Location','best');
    legend(ax3, 'True', 'EKF', 'Location','best');
    
    % y-limits: [-pi, pi] is a nice default
    ylim(ax1, [-350 350]);
    ylim(ax2, [-80 -60]);
    ylim(ax3, [-200 -120]);
    
    % x-limits: from first sample to last used eul index
    t_max = 1 + (num_frames-1)*stride;   % sample index
    xlim(ax1, [t(1) t(t_max)]);
    xlim(ax2, [t(1) t(t_max)]);
    xlim(ax3, [t(1) t(t_max)]);
    
    drawnow;
    
    % ---------- Main loop over frames ----------
    for k = 1:num_frames
    
        % eul index for this frame: 1, 11, 21, ...
        idx_eul = 1 + (k-1)*stride;
    
        % image index in img_files: img_idx, img_idx+1, ...
        img_k = image_idx + (k-1);
        img_path = fullfile(camFolder, img_files(img_k).name);
    
        % ----- Left: camera image -----
        I = imread(img_path);
        imshow(I, 'Parent', ax_img);
        title(ax_img, sprintf('Cam0 | image %d (%s)', img_k, img_files(img_k).name));
    
        % ----- Right: plot eul up to idx_eul -----
        tk = t(1:idx_eul);
    
%         yaw_unwrapped   = unwrap(eul_true(1:idx_eul,1));
        pitch_unwrapped = unwrap(eul_true(1:idx_eul,2));
        roll_unwrapped  = unwrap(eul_true(1:idx_eul,3));
%         yaw_true   = rad2deg(yaw_unwrapped);
        pitch_true = rad2deg(pitch_unwrapped);
        roll_true  = rad2deg(roll_unwrapped);
         
%         yaw_unwrapped   = unwrap(eul_ekf(1:idx_eul,1));
        pitch_unwrapped = unwrap(eul_eekf(1:idx_eul,2));
        roll_unwrapped  = unwrap(eul_eekf(1:idx_eul,3));
%         yaw_ekf   = rad2deg(yaw_unwrapped);
        pitch_eekf = rad2deg(pitch_unwrapped);
        roll_eekf  = rad2deg(roll_unwrapped);
        
        yaw_true   = eul_true_deg(1:idx_eul,1);
%         pitch_true = eul_true_deg(1:idx_eul,2);
%         roll_true  = eul_true_deg(1:idx_eul,3);

        yaw_eekf    = eul_eekf_deg(1:idx_eul,1);
%         pitch_eekf  = eul_eekf_deg(1:idx_eul,2);
%         roll_eekf   = eul_eekf_deg(1:idx_eul,3);

    
        set(h_true_yaw,   'XData', tk, 'YData', yaw_true);
        set(h_ekf_yaw,    'XData', tk, 'YData', yaw_eekf);
        set(h_true_pitch, 'XData', tk, 'YData', pitch_true);
        set(h_ekf_pitch,  'XData', tk, 'YData', pitch_eekf);
        set(h_true_roll,  'XData', tk, 'YData', roll_true);
        set(h_ekf_roll,   'XData', tk, 'YData', roll_eekf);
    
        drawnow;
    
        % write frame to video
        frame = getframe(fig);
        writeVideo(v, frame);
    end
    
    % ---------- Close video ----------
    close(v);
    disp(['Video saved as ', video_name]);

end
