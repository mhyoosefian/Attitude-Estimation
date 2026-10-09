function [mu_new, P_new, eps2_k] = EKF(mu_old, P_old, Q, R, u, y)

% Step 1: prediction 
F = F_ekf(mu_old, u);
G = G_ekf(mu_old, u);
mu_tmp = f(mu_old, u, zeros(3, 1), zeros(3, 1), zeros(3, 1));
P_tmp = F*P_old*F' + G*Q*G';

% Step 2: update
H = H_ekf(mu_tmp, u);
M = M_ekf(mu_tmp, u);
z = y - h(mu_tmp, u, zeros(3, 1));
S = H*P_tmp*H' + M*R*M';
K = P_tmp*H'/S;
mu_new = mu_tmp + K*z;
P_new = P_tmp - K*S*K';

% Step 3: normalization
L = L_ekf(mu_new);
mu_new = l(mu_new);
P_new = L*P_new*L';

% Compute eps2_k for consistency check
eps2_k = z'*inv(S)*z;