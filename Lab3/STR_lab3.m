% clean up the matlab environment
clear; clc; close all;

% run initialization of some paths and variables
init_setup;
load('lab3.mat');
% contains A, B, C, LQR_Kss, target_hover_state, clipping_distance
% C = eye(size(x))
A_k = A(1:20,1:20);
B_k = B(1:20, :);

H = 2000;

% actual state, not observable, do not reference
x(:,1) = target_hover_state;

% initial state estimate (given)
mu_x(:,1) = x(:,1);
% initial state observation (given)
y(:,1) = x(:,1);

% initial control
dx = compute_dx(target_hover_state, mu_x(:,1));
u(:,1) = LQR_Kss* dx;

% noise parameters
sigmaY = 0.5;
sigmaX = 1.4;
% sigmaY = 0;
% sigmaX = 0;
% Q = eye(21)*0.04 + diag([0,0,1.2,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.3,0,0]);
% R = eye(21)*1.6 + diag([0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);
Q = eye(21)*0.02;
R = eye(21)*1.2;
Sigma = eye(21)*0.2;

for t=2:H    
	% add noise to motion model:
    noise_F_T = randn(6,1)*sigmaX;
    
    % Simulate helicopter, do not change:
    x(:,t) = f_heli(x(:,t-1), u(:,t-1), dt, model, idx, noise_F_T);
    
    % add state observation noise
    v = randn(size(C*x(:,t)))*sigmaY;
    
    % observe noisy state
    y(:,t) = C*x(:,t) + v;    
    
    % use Kalman filter to calculate mean state
    % from mu_x(:,t-1), y(t) ,u(t-1)
    % mu_x(:,t) = y(:,t); % dumb take observation as estimate
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    mu_x(:,t) = A*mu_x(:,t-1) + B*u(:,t-1);
    Sigma = A * Sigma * transpose(A) + Q;
    mu_x(:,t) = mu_x(:,t) + Sigma*transpose(C)*inv(R + C * Sigma * transpose(C)) * (y(:,t) - C * mu_x(:,t));
    Sigma = Sigma - Sigma * transpose(C) * inv(R + C * Sigma * transpose(C)) * C * Sigma;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % LQR controller generates control for next step.  u(t-1) takes x(:,t-1)
    % to x(:,t).  do not change (only change mu_x value above)
	dx = compute_dx(target_hover_state, mu_x(:,t));
    dx(idx.ned) = max(min(dx(idx.ned), clipping_distance),-clipping_distance);
	u(:,t) = LQR_Kss* dx;   
    
end

figure; plot(x(idx.ned,:)'); legend('north', 'east', 'down'); title('Hover position');
figure; plot(x(idx.q,:)'); legend('qx', 'qy', 'qz', 'qw'); title('Hover quaternion');
figure; plot(x(idx.u_prev,:)'); legend('aileron','elevator','rudder','collective'); title('Hover trim');
time = (1:H);
%figure; plot(time, x(:,1:end), time, mu_x(:,1:end)); legend('x', 'mu_x')