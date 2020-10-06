%% Setup system env
clearvars; close all
load('good_seed.mat');rng(current_seed);
%%
% current_seed = rng; save('good_seed.mat', 'current_seed');
A = [0.7627 0.4596 0.1149 0.0198 0.0025 0.0003
-0.8994 0.7627 0.4202 0.1149 0.0193 0.0025
0.1149 0.0198 0.7652 0.4599 0.1149 0.0198
0.4202 0.1149 -0.8801 0.7652 0.4202 0.1149
0.0025 0.0003 0.1149 0.0198 0.7627 0.4596
0.0193 0.0025 0.4202 0.1149 -0.8994 0.7627];

B = [0.1199 0.4596 0.0025 0.0198 0.0000 0.0003
0.0025 0.0198 0.1199 0.4599 0.0025 0.0198]';
Cc = eye(6);

Fx = [eye(6); -eye(6)];
gx = [4 10 4 10 4 10 4 10 4 10 4 10]';
Fu = [eye(2); -eye(2)]; 
gu = [1; 1; 1; 1] * 0.5; 


R_fast = .1 * diag([1, 1]);
Q = diag([1 0 1 0 1 0]);
K_fast = -dlqr(A, B, Q, R_fast)
R_slow = 100 * diag([1, 1]);
K_slow = -dlqr(A, B, Q, R_slow)
%% Run system
JUMPS = 10;
n = 100 * JUMPS;
SAMPLES_IN_JUMP = ceil(n/JUMPS);
x0 = [3.0136 2.7106 4.0000 0.3585 3.8636 -3.1652]';
% for i = 1:JUMPS
%     Ref(:, i*SAMPLES_IN_JUMP) = x0 * (rand - 0.5) * 2;
% end

%%c
for datgen=1:10
Ref = zeros(6,n);
U_MUL = 0;
X_slow = Ref;
U_slow = zeros(2, n);
for i=1:JUMPS
    x0 = gx(1:6) .* (rand(6, 1) - 0.5)*1.3;
    [~,x, u, Ef, dE, y, xx,  yy, T] = ...
        run_system(A, B, [1 0], K_slow, nan, nan, nan, Ref(:, 1+(i-1)*SAMPLES_IN_JUMP:i*SAMPLES_IN_JUMP), 0, x0, gu);
    X_slow(:, 1+(i-1)*SAMPLES_IN_JUMP:i*SAMPLES_IN_JUMP) = x;
    U_slow(:, 1+(i-1)*SAMPLES_IN_JUMP:i*SAMPLES_IN_JUMP) = u;
end
E_train =  X_slow-Ref;
for igen=1:10
num_of_controllers = 3;
objective = [0.1164,0.9591,0.1987,0.7386,0.128,0.6054,0.02765,0.08074,0.06550 0.5782,0.09024,0.01522]; %rand(1, 12)
objectives = repmat(objective, num_of_controllers, 1);
scmc = SelfCluteringMultiController(objectives, gx(1:6), gu(1:2), []);
try
    scmc = scmc.train_on_data(E_train); 
catch
%     continue;
end
disp(['Done===========================', num2str(scmc.numOfControllers)])
%%
Ref = zeros(6, 70);
x0 = [3.0136 2.7106 4.0000 0.3585 3.8636 -3.1652]';
[~,X_slow, U_slow, Ef, dE, y, xx,  yy, T] = ...
        run_system(A, B, [1 0], K_slow, nan, nan, nan, Ref, 0, x0, gu);
perf_slow = sqrt(U_MUL * sum(sum(U_slow.^2)) + (sum(sum((X_slow - Ref).^2))));

[X_multi, U_multi, E_multi, dE, y, xx, yy, T, Ki] = run_control_system(A, B, Cc, scmc, Ref, x0, 0);
perf_mc = sqrt(U_MUL * sum(sum(U_multi.^2)) + (sum(sum((X_multi - Ref).^2))));

fprintf('Slow controller performance: %.2f  %.2f\n', perf_slow, perf_slow / perf_slow)
fprintf('Multi controller performance: %.2f   %.2f\n', perf_mc, perf_mc / perf_slow);
K_slow - scmc.controllers{1}.K

%%
if perf_mc < 27
    break;
end
% fprintf('SH controller performance: %.2f  %.2f\n', perf_fast, perf_fast / perf_slow)
end
if perf_mc < 27
    break;
end
end
%%
figure(1);clf
for i=0:1
    for j=1:3
        subplot(2, 3, i*3+j)
        hold on
        plot(T, X_slow(i*3+j, :), '--r','linewidth',2);
        plot(T, X_multi(i*3+j, :), 'b','linewidth',2);
        xlabel('k', 'FontSize', 16)
        ylabel(sprintf('x_{%d}', i*3+j), 'FontSize', 16)
    end
end
legend({'Teaching controller', 'SCMC'}, 'FontSize', 14)
save_to_multi_images(1, '../../scmc/images/ex2_result_31')