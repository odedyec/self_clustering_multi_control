%% Clear and add paths
clearvars; clc; close all;addpath(genpath('./'))
%% Setup everything

A = [1 1; 0 1];
B = [0.5; 1];
Cc = [1 0];
n = size(A, 1); 
m = size(B, 2);
Fx = [eye(2); -eye(2)];
gx = [25, 5, 25, 5]';
Fu = [eye(1); -eye(1)]; 
gu = [1; 1]; 

Q = [1 0; 0 0];

plant = Plant(A, B, gx, gu);
%%% Fast LQR controller
R_teacher = 1000; 
K_teacher = -dlqr(A, B, Q, R_teacher); 
teacher_controller = SingleController(n, m, gu, K_teacher);
%% Collect samples
x0 = [0; 0];
sim_time = 300;
ref = random_step_signal_generator(2, sim_time, sim_time/10, [-12.5 12.5; 0 0]);

figure(1); clf;
logger = run_system(plant, teacher_controller, x0, sim_time, ref);
logger.plot(1, [2; 1], 'b', 1, [0, 1, 0, 2]);
xlabel('Time[k]')
subplot(2, 1, 1);title('Teacher controller collecting data');ylabel('x_1');
subplot(2, 1, 2);ylabel('x_2')

%% Create controller
num_of_controllers = 2;
objectives = repmat([1, 0.3], num_of_controllers, 1);
scmc = SelfCluteringMultiController(objectives, gx(1:2), gu(1), []);
scmc.colors = colormap('jet'); scmc.colors = scmc.colors(ceil(linspace(1, 64, num_of_controllers)), :);
scmc = scmc.train_on_data(ref-logger.x); %, 11, xx, yy);
%% Check results
logger_scmc = run_system(plant, scmc, x0, sim_time, ref);
logger_scmc.plot(1, [2; 1], 'g', 1, [0, 1, 0, 2]);