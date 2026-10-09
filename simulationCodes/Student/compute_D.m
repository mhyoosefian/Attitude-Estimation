function [D11, D12, D21, D22_Q] = compute_D(x_old, x_new, Q, R, u, nu_t)
omega = u(1:3);
dt = u(end);
gyro_bias = x_old(5:7);
omega_bar = omega - gyro_bias;
F = [eye(3)-skew(omega_bar)*dt     -eye(3)*dt    zeros(3)
     zeros(3)                      eye(3)        zeros(3)
     zeros(3)                      zeros(3)      eye(3)];
G = blkdiag(-eye(3), eye(3), eye(3));
Gm = u(4:6);
Gg = [0;0; u(7)];
Rot = quat2rotmat(x_new(1:4));
gB = -Rot' * Gg;
mB = Rot'*Gm;
H = [skew(gB) zeros(3) eye(3)
     skew(mB) zeros(3) zeros(3)];
Qeff = G*Q*G' * dt;
inv_Qeff = inv(Qeff);
D11 = F' * inv_Qeff * F;
D12 = -F' * inv_Qeff;
D21 = D12';
R = blkdiag(R(1:3, 1:3), ((nu_t + 1)/(nu_t + 3))*eye(3));
D22_Q = inv_Qeff + H'*inv(R)*H;

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
end