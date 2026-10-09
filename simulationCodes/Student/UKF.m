function [mu_new, P_new, eps2_k] = UKF(mu_old, P_old, Q, R, u, y, alpha, beta, kappa)

n = length(mu_old);
m = length(y);
d = 9;
L = d + m + n;
lambda = alpha^2*(L+kappa) - L;
etha_m = [lambda/(L+lambda); ones(2*L,1)*1/(2*(L+lambda))];
etha_c = [lambda/(L+lambda) + 1 - alpha^2 + beta; ones(2*L,1)*1/(2*(L+lambda))];
sqrt_R = chol(R, 'lower');
sqrt_Q = chol(Q, 'lower');

Xha = [mu_old; zeros(d,1); zeros(m,1)];
sqrt_P = chol(P_old,'lower');
sqrt_Pa = [sqrt_P zeros(n,d) zeros(n,m)
           zeros(d,n) sqrt_Q zeros(d,m)
           zeros(m,n) zeros(m,d) sqrt_R];
chi1 = [Xha, Xha*ones(1,L) + sqrt(L+lambda)*sqrt_Pa, Xha*ones(1,L) - sqrt(L+lambda)*sqrt_Pa];
chi1_X = chi1(1:n,:);
chi1_W = chi1(n+1:n+d,:);
chi1_V = chi1(n+d+1:L,:);
for i=1:2*L+1
    chi2_X(:,i) = f(chi1_X(:,i), u, chi1_W(1:3,i), chi1_W(4:6,i), chi1_W(7:9,i));     % First unsceted transform
end
Xh_minus_ukf = chi2_X*etha_m;
P_minus_ukf = zeros(n,n);
for i =1:2*L+1
    P_minus_ukf = P_minus_ukf + etha_c(i)*((chi2_X(:,i) - Xh_minus_ukf)*(chi2_X(:,i) - Xh_minus_ukf).');
end
Psi = zeros(m, 2*L+1);
for i=1:2*L+1
    Psi(:,i) = h(chi2_X(:,i), u, chi1_V(1:3,i), chi1_V(4:6,i));     % Second unscented transform
end
yhat = Psi*etha_m;
Pyy = zeros(m,m);
Pxy = zeros(n,m);
for i =1:2*L+1
    Pyy = Pyy + etha_c(i)*(Psi(:,i) - yhat)*(Psi(:,i) - yhat).';
    Pxy = Pxy + etha_c(i)*(chi2_X(:,i) - Xh_minus_ukf)*(Psi(:,i) - yhat).';
end
K = Pxy/Pyy;
mu_new = Xh_minus_ukf + K*(y - yhat);
P_new = P_minus_ukf - K*Pyy*K.';
z = y - yhat;
eps2_k = z'*inv(Pyy)*z;


% Re-normalize the quaternions
Xha = mu_new;
sqrt_P = chol(P_new,'lower');
sqrt_Pa = sqrt_P;
L = n;
lambda = alpha^2*(L+kappa) - L;
etha_m = [lambda/(L+lambda); ones(2*L,1)*1/(2*(L+lambda))];
etha_c = [lambda/(L+lambda) + 1 - alpha^2 + beta; ones(2*L,1)*1/(2*(L+lambda))];
chi1 = [Xha, Xha*ones(1,L) + sqrt(L+lambda)*sqrt_Pa, Xha*ones(1,L) - sqrt(L+lambda)*sqrt_Pa];
chi1_X_norm = chi1(1:n,:);
for i=1:2*L+1
    chi2_X_norm(:,i) = l(chi1_X_norm(:,i));     % First unsceted transform
end
Xh_minus_ukf = chi2_X_norm*etha_m;
P_minus_ukf = zeros(n,n);
for i =1:2*L+1
    P_minus_ukf = P_minus_ukf + etha_c(i)*((chi2_X_norm(:,i) - Xh_minus_ukf)*(chi2_X_norm(:,i) - Xh_minus_ukf).');
end
P_new = P_minus_ukf;
mu_new = Xh_minus_ukf;
mu_new(1:4) = mu_new(1:4) / norm(mu_new(1:4));
P_new = 0.5*(P_new + P_new');
