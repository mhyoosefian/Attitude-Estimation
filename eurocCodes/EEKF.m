function [mu_new, P_new, eps2_k, R_new, eps_avg_new] = EEKF(mu_old, P_old, Q, R, u, y, t, eps_avg_old)

% Step 1: prediction 
F = F_ekf(mu_old, u);
G = G_ekf(mu_old, u);
mu_tmp = f(mu_old, u, zeros(3, 1), zeros(3, 1), zeros(3, 1));
P_tmp = F*P_old*F' + G*Q*G';

% Step 2: update
H = H_ekf(mu_tmp, u);
z = y - h(mu_tmp, u, zeros(3, 1));
S = H*P_tmp*H' + R;
K = P_tmp*H'/S;
mu_new = mu_tmp + K*z;
P_new = P_tmp - K*S*K';

% Step 3: normalization
L = L_ekf(mu_new);
mu_new = l(mu_new);
P_new = L*P_new*L';

% Compute eps2_k for consistency check
eps2_k = z'*inv(S)*z;
R_new = R;
eps_avg_new = 1/t * ((t-1)*eps_avg_old + eps2_k);


% Ensure consistency
ny = 3;
tau_u = ny + 3*sqrt(2*ny/t);
if eps_avg_new > tau_u
    tau_u_avg = t*tau_u - (t-1)*eps_avg_old;
    p = size(R,1);
    X = sdpvar(p, p, 'symmetric');
    Sigma = H*P_tmp*H' +X;
    epsc = 1e-10;
    constr = [];
    constr = [constr, X >= 0];
    Schur = Sigma - (z*z.')/tau_u_avg;
    constr = [constr, Schur >= epsc*eye(size(Schur))];
    obj = 1;
    opts = sdpsettings('solver','mosek','verbose',0);
    sol  = optimize(constr, obj, opts);
%     disp(sol.info)
    Xopt = value(X);
    R_new = Xopt;

    % Step 2: update
    S = value(Sigma);
    K = P_tmp*H'/S;
    mu_new = mu_tmp + K*z;
    P_new = P_tmp - K*S*K';
    
    % Step 3: normalization
    L = L_ekf(mu_new);
    mu_new = l(mu_new);
    P_new = L*P_new*L';
    eps2_k = z'*inv(S)*z;
    eps_avg_new = 1/t * ((t-1)*eps_avg_old + eps2_k);
end