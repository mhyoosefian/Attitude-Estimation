function [mu_new, P_new, eps2_k] = ESEKF(mu_old, P_old, Q, R, u, y)

% Step 1: prediction 
omega = u(1:3);
dt = u(end);
b_g_old = mu_old(5:7);
omega_bar = omega - b_g_old;
F = [eye(3) - skew(omega_bar)*dt   -eye(3)*dt    zeros(3)
     zeros(3)                      eye(3)        zeros(3)
     zeros(3)                      zeros(3)      eye(3)];
G = blkdiag(-eye(3), eye(3), eye(3));
mu_tmp = f(mu_old, u, zeros(3, 1), zeros(3, 1), zeros(3, 1));
q_tmp = mu_tmp(1:4);
b_g_tmp = mu_tmp(5:7);
b_a_tmp = mu_tmp(8:end);
P_tmp = F*P_old*F' + G*Q*G'*dt;

% Step 2: update
Gm = u(4:6);
Gg = [0;0; u(7)];
Rot = quat2rotmat(q_tmp);
gB = -Rot' * Gg;
mB = Rot' *Gm;
H = [skew(gB) zeros(3) eye(3)
     skew(mB) zeros(3) zeros(3)];
z = y - h(mu_tmp, u, zeros(3, 1), zeros(3, 1));
S = H*P_tmp*H' + R;
K = (P_tmp*H')/S;
P_new = P_tmp - K*S*K';
delX = K*z;

% Step 3: injection and normalization
delTheta = delX(1:3);
q_new = quat_prod(q_tmp , exp_q(delTheta/2));
q_new = q_new / norm(q_new);
b_g_new = b_g_tmp + delX(4:6);
b_a_new = b_a_tmp + delX(7:9);
mu_new = [q_new ; b_g_new ; b_a_new];

% Compute eps2_k for consistency check
eps2_k = z'*inv(S)*z;


% Helper functions
    function S = skew(v)
    S = [  0   -v(3)  v(2);
         v(3)   0    -v(1);
        -v(2)  v(1)   0  ];
    end
    function R = quat2rotmat(q)
        qw = q(1); q1 = q(2); q2 = q(3); q3 = q(4);
        R = [qw*qw + q1*q1 - q2*q2 - q3*q3,  2*(q1*q2 - qw*q3),               2*(q1*q3 + qw*q2);
             2*(q1*q2 + qw*q3),              qw*qw - q1*q1 + q2*q2 - q3*q3,   2*(q2*q3 - qw*q1);
             2*(q1*q3 - qw*q2),              2*(q2*q3 + qw*q1),               qw*qw - q1*q1 - q2*q2 + q3*q3];
    end
    function quat = exp_q(eta)
        quat = [cos(norm(eta)); eta/norm(eta)*sin(norm(eta))];
    end
    function out = quat_prod(p,q)
        pw = p(1);
        pv = p(2:4);
        qw = q(1);
        qv = q(2:4);
        out = [pw*qw - dot(pv,qv); pw*qv + qw*pv + cross(pv, qv)];
    end
end