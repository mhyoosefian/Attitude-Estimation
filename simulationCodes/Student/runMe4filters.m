clc
clear
close

%% Initialization

addpath utils/
N = 200;                        % time steps
numTrials = 500;
n = 9;
inpuDim = 8;
outputDim = 6;
dt = 1/20;                      % sampling time
g_ref = 9.81;
dec = deg2rad(-15);
inc = deg2rad(60);
m_ref = [cos(inc)*cos(dec);     % reference magnetic field
        cos(inc)*sin(dec);
        sin(inc)];
m_ref = m_ref / norm(m_ref);

% Noise statistics
processNoiseDim = 9;
sig_g = 0.1;                   % gyro noise std
sig_b_g = 0.01;                % gyro bias noise std
sig_a = 0.2;                   % accel noise std
sig_b_a = 0.01;                % accel bias noise std
nu_t = 3;
sig_t = sqrt(nu_t / (nu_t - 2));
Q = blkdiag(sig_g^2*eye(3) , sig_b_g^2*eye(3) , sig_b_a^2*eye(3));
sqrtQ = sqrtm(Q);
R = blkdiag(sig_a^2*eye(3) , sig_t^2*eye(3));
sqrtR = sqrtm(R);
P0 = 0.1*eye(processNoiseDim);
mu0 = [1;0;0;0; 0.2;-0.3;0.7; 0;0.4;-0.9];

% ESEKF intialization
mu_esekf(:, 1) = mu0;
P_esekf = cell(numTrials, N+1);

% EKF intialization
mu_ekf(:, 1) = mu0;
P_ekf = cell(numTrials, N+1);

% UKF initialization
stateDim = 10;
mu_ukf(:, 1) = mu0;
P_ukf = cell(numTrials, N+1);
d = 9;
L = d + outputDim + stateDim;
alpha = 1;                   % primary scaling parameter
beta = 2;                       % secondary scaling parameter
kappa = 0;                      % teritary scaling parameter

% EEKF intialization
mu_eekf(:, 1) = mu0;
P_eekf = cell(numTrials, N+1);

%% Main code

D11 = cell(numTrials, N);
D12 = cell(numTrials, N);
D21 = cell(numTrials, N);
D22 = cell(numTrials, N);
xhat_esekf = cell(numTrials, N+1);
xhat_ekf = cell(numTrials, N+1);
xhat_ukf = cell(numTrials, N+1);
xhat_eekf = cell(numTrials, N+1);
eps2_esekf = zeros(numTrials, N);
eps2_ekf = zeros(numTrials, N);
eps2_ukf = zeros(numTrials, N);
eps2_eekf = zeros(numTrials, N);
error_esekf = cell(numTrials, N);
error_ekf = cell(numTrials, N);
error_ukf = cell(numTrials, N);
error_eekf = cell(numTrials, N);
runtime_esekf = zeros(numTrials, N);
runtime_ekf = zeros(numTrials, N);
runtime_ukf = zeros(numTrials, N);
runtime_eekf = zeros(numTrials, N);
xtrue = cell(numTrials, N);
for trialIter = 1:numTrials
    disp(['iteration ', num2str(trialIter), ' out of ', num2str(numTrials)]);

    % Initial condition
    rng(trialIter, "twister");
    mu_esekf(:, 1) = mu0;
    mu_ekf(:, 1) = mu0;
    mu_ukf(:, 1) = mu0;
    mu_eekf(:, 1) = mu0;
    P_esekf{trialIter, 1} = P0;
    P_ekf{trialIter, 1} = blkdiag(0.1, P0);
    P_ukf{trialIter, 1} = blkdiag(0.1, P0);
    P_eekf{trialIter, 1} = blkdiag(0.1, P0);
    xhat_esekf{trialIter, 1} = mu0;
    xhat_ekf{trialIter, 1} = mu0;
    xhat_ukf{trialIter, 1} = mu0;
    xhat_eekf{trialIter, 1} = mu0;
    x0_tmp = sqrtm(P0)*randn(processNoiseDim, 1);
    error_esekf{trialIter, 1} = [log_q(mu0(1:4)); mu0(5:end)];
    error_ekf{trialIter, 1} = [log_q(mu0(1:4)); mu0(5:end)];
    error_ukf{trialIter, 1} = [log_q(mu0(1:4)); mu0(5:end)];
    error_eekf{trialIter, 1} = [log_q(mu0(1:4)); mu0(5:end)];
    q0 = exp_q(x0_tmp(1:3));
    x0 = mu0 + [q0; x0_tmp(4:end)];
    x0(1:4) = x0(1:4) / norm(x0(1:4));
    x(:, 1) = x0;
    xtrue{trialIter, 1} = x(:, 1);
    eps_avg = 0;

    % Measurement noise
    v(1:3, :) = sqrtR(1:3, 1:3)*randn(outputDim/2, N+1);
    v(4:6, :) = trnd(nu_t, outputDim/2, N+1);
    
    % Process noise
    w = sqrtQ*randn(processNoiseDim, N+1);

    % Simulate the system and estimators
    for t = 1:N

        % Simulate the system
        u = [0.10*sin(0.3*t) + 0.20*cos(0.2*t);
             0.30*cos(0.2*t) + 0.50*sin(0.4*t);
             0.20 + 0.50*sin(0.1*t);
             ];
        u = [u; m_ref; g_ref; dt];
        x(:, t+1) = f(x(:, t), u, w(1:3, t), w(4:6, t), w(7:9, t));
        xtrue{trialIter, t+1} = x(:, t+1);
        q_true  = x(1:4,  t+1);
        bg_true = x(5:7,  t+1);
        ba_true = x(8:10, t+1);

        % Recieve a measurement
        y(:, t+1) = h(x(:, t+1), u, v(1:3, t), v(4:6, t));
        
        % ESEKF
        tic;
        [mu_esekf(:, t+1), P_esekf{trialIter, t+1}, eps2_esekf(trialIter, t)] = ESEKF(mu_esekf(:, t), P_esekf{trialIter, t}, Q, R, u, y(:, t+1));
        runtime_esekf(trialIter, t) = toc;
        xhat_esekf{trialIter, t+1} = mu_esekf(:, t+1);
        q_hat_esekf   = mu_esekf(1:4,  t+1);
        bg_hat_esekf  = mu_esekf(5:7,  t+1);
        ba_hat_esekf  = mu_esekf(8:10, t+1);
        dq_esekf = quat_prod(quatconj(q_hat_esekf')', q_true);
        deltheta_esekf = log_q(dq_esekf);
        delbg_esekf = bg_true - bg_hat_esekf;
        delba_esekf = ba_true - ba_hat_esekf;
        error_esekf{trialIter, t+1} = [deltheta_esekf; delbg_esekf; delba_esekf];

        % EKF
        tic;
        [mu_ekf(:, t+1), P_ekf{trialIter, t+1}, eps2_ekf(trialIter, t)] = EKF(mu_ekf(:, t), P_ekf{trialIter, t}, Q, R, u, y(:, t+1));
        runtime_ekf(trialIter, t) = toc;
        xhat_ekf{trialIter, t+1} = mu_ekf(:, t+1);
        q_hat_ekf   = mu_ekf(1:4,  t+1);
        bg_hat_ekf  = mu_ekf(5:7,  t+1);
        ba_hat_ekf  = mu_ekf(8:10, t+1);
        dq_ekf = quat_prod(quatconj(q_hat_ekf')', q_true);
        deltheta_ekf = log_q(dq_ekf);
        delbg_ekf = bg_true - bg_hat_ekf;
        delba_ekf = ba_true - ba_hat_ekf;
        error_ekf{trialIter, t+1} = [deltheta_ekf; delbg_ekf; delba_ekf];

        % UKF
        tic;
        [mu_ukf(:, t+1), P_ukf{trialIter, t+1}, eps2_ukf(trialIter, t)] = UKF(mu_ukf(:, t), P_ukf{trialIter, t}, Q, R, u, y(:, t+1), alpha, beta, kappa);
        runtime_ukf(trialIter, t) = toc;
        xhat_ukf{trialIter, t+1} = mu_ukf(:, t+1);
        q_hat_ukf   = mu_ukf(1:4,  t+1);
        bg_hat_ukf  = mu_ukf(5:7,  t+1);
        ba_hat_ukf  = mu_ukf(8:10, t+1);
        dq_ukf = quat_prod(quatconj(q_hat_ukf')', q_true);
        deltheta_ukf = log_q(dq_ukf);
        delbg_ukf = bg_true - bg_hat_ukf;
        delba_ukf = ba_true - ba_hat_ukf;
        error_ukf{trialIter, t+1} = [deltheta_ukf; delbg_ukf; delba_ukf];

        % EEKF
        tic;
        [mu_eekf(:, t+1), P_eekf{trialIter, t+1}, eps2_eekf(trialIter, t), ~, eps_avg] = EEKF(mu_eekf(:, t), P_eekf{trialIter, t}, Q, R, u, y(:, t+1), t, eps_avg);
        runtime_eekf(trialIter, t) = toc;
        xhat_eekf{trialIter, t+1} = mu_eekf(:, t+1);
        q_hat_eekf   = mu_eekf(1:4,  t+1);
        bg_hat_eekf  = mu_eekf(5:7,  t+1);
        ba_hat_eekf  = mu_eekf(8:10, t+1);
        dq_eekf = quat_prod(quatconj(q_hat_eekf')', q_true);
        deltheta_eekf = log_q(dq_eekf);
        delbg_eekf = bg_true - bg_hat_eekf;
        delba_eekf = ba_true - ba_hat_eekf;
        error_eekf{trialIter, t+1} = [deltheta_eekf; delbg_eekf; delba_eekf];

        % Compute Matrices for PCRB
        [D11{trialIter, t}, D12{trialIter, t}, D21{trialIter, t}, D22{trialIter, t}] = compute_D(x(:, t), x(:, t+1), Q, R, u, nu_t);
    end
end

% Compute PCRB and MSE
J = cell(1, N);
PCRB = cell(1, N);
MSE_esekf = cell(1, N);
MSE_ekf = cell(1, N);
MSE_ukf = cell(1, N);
MSE_eekf = cell(1, N);
E_P_esekf = cell(1, N);
E_P_ekf = cell(1, N);
E_P_ukf = cell(1, N);
E_P_eekf = cell(1, N);
J{1} = inv(P0);
PCRB{1} = P0;
eta_esekf = zeros(1, N);
eta_ekf = zeros(1, N);
eta_ukf = zeros(1, N);
eta_eekf = zeros(1, N);
for k = 1:N

    % Compute PCRB 
    E_D11_k = mean(cat(3, D11{:, k}), 3);
    E_D12_k = mean(cat(3, D12{:, k}), 3);
    E_D21_k = mean(cat(3, D21{:, k}), 3);
    E_D22_k = mean(cat(3, D22{:, k}), 3);
    J{k+1} = E_D22_k - E_D21_k * inv(J{k} + E_D11_k) * E_D12_k;
    PCRB{k+1} = inv(J{k+1});
end
for k = 1:N+1

    % Compute expectation of filter covariance
    E_P_esekf{k} = mean(cat(3, P_esekf{:, k}), 3); 
    E_P_ekf{k} = mean(cat(3, P_ekf{:, k}), 3); 
    E_P_ukf{k} = mean(cat(3, P_ukf{:, k}), 3);
    E_P_eekf{k} = mean(cat(3, P_ekf{:, k}), 3); 
    
    % Compute MSE
    MSE_matrix_esekf = zeros(n, n);
    MSE_matrix_ekf = zeros(n, n);
    MSE_matrix_ukf = zeros(n, n);
    MSE_matrix_eekf = zeros(n, n);
    for trialIter = 1:numTrials
        error_vector_esekf = error_esekf{trialIter, k};
        MSE_matrix_esekf = MSE_matrix_esekf + error_vector_esekf * error_vector_esekf';
        error_vector_ekf = error_ekf{trialIter, k};
        MSE_matrix_ekf = MSE_matrix_ekf + error_vector_ekf * error_vector_ekf';
        error_vector_ukf = error_ukf{trialIter, k};
        MSE_matrix_ukf = MSE_matrix_ukf + error_vector_ukf * error_vector_ukf';
        error_vector_eekf = error_eekf{trialIter, k};
        MSE_matrix_eekf = MSE_matrix_eekf + error_vector_eekf * error_vector_eekf';
    end
    MSE_esekf{k} = MSE_matrix_esekf / numTrials;
    MSE_ekf{k} = MSE_matrix_ekf / numTrials;
    MSE_ukf{k} = MSE_matrix_ukf / numTrials;
    MSE_eekf{k} = MSE_matrix_eekf / numTrials;

    % Compute eta
    eta_esekf(k) = trace(PCRB{k}) ./ trace(MSE_esekf{k});
    eta_ekf(k) = trace(PCRB{k}) ./ trace(MSE_ekf{k});
    eta_ukf(k) = trace(PCRB{k})  ./ trace(MSE_ukf{k});
    eta_eekf(k) = trace(PCRB{k}) ./ trace(MSE_eekf{k});
end

% Compute average epsilon (NIS)
eps_avg_esekf = cumsum(eps2_esekf, 2) ./ (1:N);
eps_avg_ekf = cumsum(eps2_ekf, 2) ./ (1:N);
eps_avg_ukf = cumsum(eps2_ukf, 2) ./ (1:N);
eps_avg_eekf = cumsum(eps2_eekf, 2) ./ (1:N);
std_eps_avg_esekf = std(eps_avg_esekf, 0, 1);
std_eps_avg_ekf = std(eps_avg_ekf, 0, 1);
std_eps_avg_ukf = std(eps_avg_ukf, 0, 1);
std_eps_avg_eekf = std(eps_avg_eekf, 0, 1);
mean_eps_avg_esekf = mean(eps_avg_esekf, 1);
mean_eps_avg_ekf = mean(eps_avg_ekf, 1);
mean_eps_avg_ukf = mean(eps_avg_ukf, 1);
mean_eps_avg_eekf = mean(eps_avg_eekf, 1);
mean_eps_avg_true = outputDim*ones(N,1);
std_eps_avg_true = sqrt(2*outputDim ./ (1:N));

% Compute runtime
cpu_time_esekf = geo_mean(mean(runtime_esekf, 2));
cpu_time_ekf = geo_mean(mean(runtime_ekf, 2));
cpu_time_ukf = geo_mean(mean(runtime_ukf, 2));
cpu_time_eekf = geo_mean(mean(runtime_eekf, 2));

% Compute angle MSE (for both UKF and ES-EKF)
MSE_angles_esekf = cell(1, N);
MSE_angles_ekf = cell(1, N);
MSE_angles_ukf = cell(1, N);
MSE_angles_eekf = cell(1, N);
for k = 1:N+1

    % Compute MSE
    MSE_matrix_esekf = zeros(3, 3);
    MSE_matrix_ekf = zeros(3, 3);
    MSE_matrix_ukf = zeros(3, 3);
    MSE_matrix_eekf = zeros(3, 3);
    for trialIter = 1:numTrials

        % Quaternions -> Euler (yaw, pitch, roll) in radians
        eul_esekf  = quat2eul(xhat_esekf{trialIter, k}(1:4)');   % 1x3
        eul_ekf  = quat2eul(xhat_ekf{trialIter, k}(1:4)');
        eul_ukf  = quat2eul(xhat_ukf{trialIter, k}(1:4)');
        eul_eekf  = quat2eul(xhat_eekf{trialIter, k}(1:4)');
        eul_true = quat2eul(xtrue{trialIter, k}(1:4)');

        % Wrap angle errors to [-pi, pi]
        ang_err_esekf = wrapToPi(eul_esekf  - eul_true);   % 1x3
        ang_err_ekf = wrapToPi(eul_ekf  - eul_true);  
        ang_err_ukf = wrapToPi(eul_ukf  - eul_true); 
        ang_err_eekf = wrapToPi(eul_eekf  - eul_true);  

        MSE_matrix_esekf = MSE_matrix_esekf + ang_err_esekf(:) * ang_err_esekf(:).';
        MSE_matrix_ekf = MSE_matrix_ekf + ang_err_ekf(:) * ang_err_ekf(:).';
        MSE_matrix_ukf = MSE_matrix_ukf + ang_err_ukf(:) * ang_err_ukf(:).';
        MSE_matrix_eekf = MSE_matrix_eekf + ang_err_eekf(:) * ang_err_eekf(:).';
    end
    MSE_angles_esekf{k} = MSE_matrix_esekf / numTrials;
    MSE_angles_ekf{k} = MSE_matrix_ekf / numTrials;
    MSE_angles_ukf{k} = MSE_matrix_ukf / numTrials;
    MSE_angles_eekf{k} = MSE_matrix_eekf / numTrials;
end

%% Plot the results

colors = {[0.6350, 0.0780, 0.1840], [0.4660, 0.6740, 0.1880], [0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980]};
lineStyles = {'-', '-.'};
% Plot the RMSE and PCRB
folderName = 'figs';
if ~exist(folderName, 'dir')
       mkdir(folderName)
end
yLabels = {'RMSE ($\delta \theta_x$)' , 'RMSE ($\delta \theta_y$)' , 'RMSE ($\delta \theta_z$)' , 'RMSE ($b_{g_x}$)' , 'RMSE ($b_{g_y}$)' , 'RMSE ($b_{g_z}$)', ...
    'RMSE ($b_{a_x}$)' , 'RMSE ($b_{a_y}$)' , 'RMSE ($b_{a_z}$)'};
for i=1:n
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
    tmp = cat(3, MSE_esekf{:});
    rmse_esekf = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
    if i > 3
        tmp = cat(3, MSE_ekf{:});
        rmse_ekf = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
        tmp = cat(3, MSE_ukf{:});
        rmse_ukf = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
        tmp = cat(3, MSE_eekf{:});
        rmse_eekf = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
    end
    tmp = cat(3, PCRB{:});
    rmse_bound = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
    tmp = cat(3, E_P_esekf{:});
    std_esekf = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
    if i > 3
        tmp = cat(3, E_P_ekf{:});
        std_ekf = reshape(sqrt(tmp(i+1, i+1, :)), [N+1, 1]);
        tmp = cat(3, E_P_ukf{:});
        std_ukf = reshape(sqrt(tmp(i+1, i+1, :)), [N+1, 1]);
        tmp = cat(3, E_P_eekf{:});
        std_eekf = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
    end
    plot(0:N, rmse_esekf, 'Color', colors{1});
    plot(0:N, std_esekf, 'Color', colors{1}, 'LineStyle', lineStyles{2});
    if i > 3
        plot(0:N, rmse_ekf, 'Color', colors{2});
        plot(0:N, std_ekf, 'Color', colors{2}, 'LineStyle', lineStyles{2});
        plot(0:N, rmse_ukf, 'Color', colors{3});
        plot(0:N, std_ukf, 'Color', colors{3}, 'LineStyle', lineStyles{2});
        plot(0:N, rmse_eekf, 'Color', colors{4});
        plot(0:N, std_eekf, 'Color', colors{4}, 'LineStyle', lineStyles{2});
    end
    plot(0:N, rmse_bound, 'Color', 'k');
    axis tight;
    xlabel('Time step')
    ylabel(yLabels{i})
    ax1 = gca;

    % Plot dummy figures for legends
    h1 = plot(ax1, NaN, NaN, 'o', 'Color', colors{1});
    if i > 3 
        h2 = plot(ax1, NaN, NaN, 'o', 'Color', colors{2});
        h3 = plot(ax1, NaN, NaN, 'o', 'Color', colors{3});
        h4 = plot(ax1, NaN, NaN, 'o', 'Color', colors{4});
    end
    h5 = plot(ax1, NaN, NaN, 'o', 'Color', 'k');
    if i > 3
        leg1 = legend(ax1, [h1, h2, h3, h4, h5], {'ESKF', 'EKF', 'UKF', 'EEKF', '$\sqrt{\mathrm{PCRB}}$'}, ...
            'Location', 'northoutside', 'Box', 'off', 'Orientation', 'horizontal');
    else 
        leg1 = legend(ax1, [h1, h5], {'ESKF', '$\sqrt{\mathrm{PCRB}}$'}, ...
            'Location', 'northoutside', 'Box', 'off', 'Orientation', 'horizontal');
    end
    leg1.FontSize = 8;
    ax2 = axes('Position', ax1.Position, ...
           'Color', 'none', ...
           'XLim', ax1.XLim, ...
           'YLim', ax1.YLim, ...
           'XTick', [], 'YTick', [], ...
           'HitTest', 'off', ...      % so clicks still hit ax1
           'Box', 'off');
    hold(ax2, 'on');
    h6 = plot(NaN, NaN, 'k', 'LineStyle', lineStyles{1}, 'LineWidth', 1.5);
    h7 = plot(NaN, NaN, 'k', 'LineStyle', lineStyles{2}, 'LineWidth', 1.5);
    leg2 = legend(ax2, [h6, h7], {'RMSE', 'Std'}, 'Location', 'best', 'Box', 'on', 'Orientation', 'vertical');
    axis tight;

    % Save the figure
    fileName = sprintf('fig_RMSE_%d_student.pdf', i);
    filePath = fullfile(folderName, fileName);
    exportgraphics(f, filePath, 'ContentType', 'vector');
end

% Plot eta
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
plot(0:N, eta_esekf, 'color', colors{1});
plot(0:N, eta_ekf, 'color', colors{2});
plot(0:N, eta_ukf, 'color', colors{3});
plot(0:N, eta_eekf, 'color', colors{4});
ylabel('Filters efficiency ($\eta$)')
xlabel('Time step')
lgd = legend('ESEKF', 'EKF', 'UKF', 'EEKF', 'Location', 'northoutside', 'Orientation', 'horizontal');
lgd.Box = 'off';
axis tight;
fileName = 'fig_eta_student.pdf';
filePath = fullfile(folderName, fileName);
exportgraphics(f, filePath, 'ContentType', 'vector');


% Plot average epsilon for ESEKF
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
plot(1:N, mean_eps_avg_esekf, 'color', colors{1});
fill([1:N N:-1:1], [mean_eps_avg_esekf+z*std_eps_avg_esekf  mean_eps_avg_esekf(end:-1:1)-z*std_eps_avg_esekf(end:-1:1)], colors{1}, ...
        'FaceAlpha', faceAlpha, 'HandleVisibility', 'off', 'EdgeColor', 'none');
plot(1:N, mean_eps_avg_true, 'color', 'k');
plot(1:N, mean_eps_avg_true + z*std_eps_avg_true', 'Color', 'k', 'LineStyle', '-.');
plot(1:N, mean_eps_avg_true - z*std_eps_avg_true', 'Color', 'k', 'LineStyle', '-.');
ylabel('$\bar \epsilon^2_k$')
xlabel('Time step')
title('ESEKF');
ylim([-2 12]);
fileName = 'fig_eps2ESKF_student.pdf';
filePath = fullfile(folderName, fileName);
exportgraphics(f, filePath, 'ContentType', 'vector');


% Plot average epsilon for EKF
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
plot(1:N, mean_eps_avg_ekf, 'color', colors{2});
fill([1:N N:-1:1], [mean_eps_avg_ekf+z*std_eps_avg_ekf  mean_eps_avg_ekf(end:-1:1)-z*std_eps_avg_ekf(end:-1:1)], colors{2}, ...
        'FaceAlpha', faceAlpha, 'HandleVisibility', 'off', 'EdgeColor', 'none');
plot(1:N, mean_eps_avg_true, 'color', 'k');
plot(1:N, mean_eps_avg_true + z*std_eps_avg_true', 'Color', 'k', 'LineStyle', '-.');
plot(1:N, mean_eps_avg_true - z*std_eps_avg_true', 'Color', 'k', 'LineStyle', '-.');
ylabel('$\bar \epsilon^2_k$')
xlabel('Time step')
title('EKF');
ylim([-2 12]);
fileName = 'fig_eps2EKF_student.pdf';
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
plot(1:N, mean_eps_avg_ukf, 'color', colors{3});
fill([1:N N:-1:1], [mean_eps_avg_ukf+z*std_eps_avg_ukf  mean_eps_avg_ukf(end:-1:1)-z*std_eps_avg_ukf(end:-1:1)], colors{3}, ...
        'FaceAlpha', faceAlpha, 'HandleVisibility', 'off', 'EdgeColor', 'none');
plot(1:N, mean_eps_avg_true, 'color', 'k');
plot(1:N, mean_eps_avg_true + z*std_eps_avg_true', 'Color', 'k', 'LineStyle', '-.');
plot(1:N, mean_eps_avg_true - z*std_eps_avg_true', 'Color', 'k', 'LineStyle', '-.');
ylabel('$\bar \epsilon^2_k$')
xlabel('Time step')
title('UKF');
ylim([-2 12]);
fileName = 'fig_eps2UKF_student.pdf';
filePath = fullfile(folderName, fileName);
exportgraphics(f, filePath, 'ContentType', 'vector');


% Plot average epsilon for EEKF
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
plot(1:N, mean_eps_avg_eekf, 'color', colors{4});
fill([1:N N:-1:1], [mean_eps_avg_eekf+z*std_eps_avg_eekf  mean_eps_avg_eekf(end:-1:1)-z*std_eps_avg_eekf(end:-1:1)], colors{4}, ...
        'FaceAlpha', faceAlpha, 'HandleVisibility', 'off', 'EdgeColor', 'none');
plot(1:N, mean_eps_avg_true, 'color', 'k');
plot(1:N, mean_eps_avg_true + z*std_eps_avg_true', 'Color', 'k', 'LineStyle', '-.');
plot(1:N, mean_eps_avg_true - z*std_eps_avg_true', 'Color', 'k', 'LineStyle', '-.');
ylabel('$\bar \epsilon^2_k$')
xlabel('Time step')
title('EEKF');
ylim([-2 12]);
fileName = 'fig_eps2EEKF_student.pdf';
filePath = fullfile(folderName, fileName);
exportgraphics(f, filePath, 'ContentType', 'vector');

% Show cpu time
disp(['average CPU time for ESEKF is ' , num2str(cpu_time_esekf), ' (s)']);
disp(['average CPU time for EKF is ' , num2str(cpu_time_ekf), ' (s)']);
disp(['average CPU time for UKF is ' , num2str(cpu_time_ukf), ' (s)']);
disp(['average CPU time for EEKF is ' , num2str(cpu_time_eekf), ' (s)']);

% Plot angle RMSE
for i=1:3
    tmp = cat(3, MSE_angles_esekf{:});
    rmse_angles_esekf = reshape(sqrt(tmp(i, i, :)), [N+1, 1]);
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
    plot(0:N, (180/pi).*rmse_angles_esekf, 'Color', colors{1});
    plot(0:N, (180/pi).*rmse_angles_ekf, 'Color', colors{2});
    plot(0:N, (180/pi).*rmse_angles_ukf, 'Color', colors{3});
    plot(0:N, (180/pi).*rmse_angles_eekf, 'Color', colors{4});
    lgd = legend('ESEKF', 'EKF', 'UKF', 'EEKF', 'Location', 'northoutside', 'Orientation', 'horizontal');
    lgd.Box = 'off';
    if i == 1
        ylabel('RMSE of $\psi (^\circ)$')
    elseif i == 2
        ylabel('RMSE of $\theta (^\circ)$')
    else
        ylabel('RMSE of $\phi (^\circ)$')
    end
    xlabel('Time step')
    axis tight;
    fileName = sprintf('fig_RMSE_angles_%d_student.pdf', i);
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

%% Helper function
function quat = exp_q(eta)
    quat = [cos(norm(eta)); eta/norm(eta)*sin(norm(eta))];
end
function dtheta = log_q(q)
    q = q(:) / norm(q);  % normalize
    qw = q(1);
    qv = q(2:4);
    
    % Clamp qw to avoid NaNs due to rounding errors
    qw = min(max(qw, -1.0), 1.0);
    
    theta = 2 * acos(qw);
    s = sqrt(1 - qw^2);
    
    if s < 1e-8
        % Small angle: use linear approximation
        dtheta = 2 * qv;
    else
        dtheta = (theta / s) * qv;
    end
end
function out = quat_prod(p,q)
    pw = p(1);
    pv = p(2:4);
    qw = q(1);
    qv = q(2:4);
    out = [pw*qw - dot(pv,qv); pw*qv + qw*pv + cross(pv, qv)];
end