%% Setup system env
clearvars; close all
%%
% current_seed = rng; save('good_seed.mat', 'current_seed');
A = [1 1; 0 1];
B = [0.5; 1];
rank([B A *B])
Cc = [1 0];
f = @(t, X, u)(A * X + B * u);

Fx = [eye(2); -eye(2)];
gx = [25, 5, 25, 5]';
Fu = [eye(1); -eye(1)]; 
gu = [1; 1]; 


R = 1.10;
Q = [1 0; 0 1];
K_fast = -dlqr(A, B, Q, R);
R = 1000;
K_slow = -dlqr(A, B, Q, R);

JUMPS = 100;
n = 30 * JUMPS;
Ref = zeros(2,n);
SAMPLES_IN_JUMP = ceil(n/JUMPS);
for i = 1:JUMPS
    Ref(1, 1+(i-1)*SAMPLES_IN_JUMP:i*SAMPLES_IN_JUMP) = ones(1, SAMPLES_IN_JUMP) * (rand - 0.5) * 2 * 20;
end
x0 = [0;0];
%% Run system
U_MUL = 0;
[~,X_slow, U_slow, Ef, dE, y, xx,  yy, T] = ...
    run_system(A, B, [1 0], K_slow, nan, nan, nan, Ref, 0, x0, gu);
perf_slow = U_MUL * sum(U_slow * U_slow') + (sum(sum((X_slow - Ref).^2)));
% figure(1); clf;draw_omega2(A, B, K_slow,  Fx, Fu, gx, gu, 1, 'r')
% hold on; sh.draw(1); hold off
% disp('Done')
figure(2); plot(T, X_slow(1, :), T, Ref(1, :)); xlabel('Time');ylabel('x_1(t), r(t)');legend('x_1', 'r')
figure(3); plot(T(1:60), X_slow(1, 1:60), T(1:60), Ref(1, 1:60)); xlabel('Time');ylabel('x_1(t), r(t)');legend('x_1', 'r')
% save_to_multi_images(3, '../../scmc/images/slow_60sec')
% save_to_multi_images(2, '../../scmc/images/slow')
% K_sh = get_k_from_convex_points(sh.ConvexHull, 1)
% K_sh ./ K_slow
%%
num_of_controllers = 2;
E_slow =  X_slow-Ref;
figure(4);quiver(E_slow(1, :), E_slow(2, :), xx, yy, 'AutoScale', 'off', 'HandleVisibility','off');xlabel('e_1'); ylabel('e_2');
% save_to_multi_images(4, '../../scmc/images/dataset')
% objectives = [1, 0.5;1, 0.5;1, 0.5;1, 0.5];
objectives = repmat([1, 0.3], num_of_controllers, 1);
scmc = SelfCluteringMultiController(objectives, gx(1:2), gu(1), []);
scmc.colors = colormap('jet'); scmc.colors = scmc.colors(ceil(linspace(1, 64, num_of_controllers)), :);
scmc = scmc.train_on_data(E_slow, 11, xx, yy);
%%
% save_to_multi_images(11, '../../scmc/images/scmc')
% save_to_multi_images(12, '../../scmc/images/level_sets')
real_actual=82;figure(real_actual); clf;scmc.drawAll(real_actual, true);
scmc.drawActual(A, B, Fx, Fu, gx, gu, real_actual);legend('Estimated MAS 1', 'Estimated MAS 2', 'Actual MAS 2', 'Actual MAS 1')
for i=1:scmc.numOfControllers
    fprintf('K%d: %s     K%d./K_slow: %s\n', i, num2str(scmc.controllers{i}.K), i, num2str(scmc.controllers{i}.K ./ K_slow));
end
[X_multi, U_multi, E_multi, dE, y, xx, yy, T, Ki] = run_control_system(A, B, Cc, scmc, Ref, x0, 0);
% [~,X_sh, U_sh, Ef, dE, y, ~, ~, T] = ...
%     run_system(A, B, [1 0], K_sh, nan, nan, nan, Ref, 0, x0, gu);
% X_sh = X_slow;
% U_sh = U_slow;
% perf_fast = U_MUL * sum(sum(U_sh.^2)) + sqrt(sum(sum((X_sh - Ref).^2)));

figure(14)
subplot(5, 1, 1);plot(T, Ki, '.'); axis([0 60 0 scmc.numOfControllers]);yticks(1:scmc.numOfControllers);ylabel('K_i')
subplot(5, 1, 2:3);plot(T, X_multi(1, :), T, X_slow(1, :), T, Ref(1, :));legend('SCMC', 'Slow', 'Ref');xlim([0 60]);ylabel('x_1')
subplot(5, 1, 4:5);plot(T, U_multi, T, U_slow);legend('SCMC', 'Slow');axis([0 60 -1 1]);ylabel('u');xlabel('Time')
% save_to_multi_images(14, '../../scmc/images/scmc_60_sec')

figure(15);clf
scmc.drawAll(15);
% save_to_multi_images(15, '../../scmc/images/sets_no_data')
T_end = 29;for i=1:T_end;quiver(E_multi(1, i), E_multi(2, i), xx(i), yy(i), 'Color', scmc.colors(Ki(i), :), 'HandleVisibility','off');end
% save_to_multi_images(15, '../../scmc/images/sets_w_new_data')

fprintf('Slow controller performance: %.2f  %.2f\n', perf_slow, perf_slow / perf_slow)
perf_mc = U_MUL * sum(sum(U_multi .^ 2)) + (sum(sum((X_multi - Ref).^2)));
fprintf('Multi controller performance: %.2f   %.2f\n', perf_mc, perf_mc / perf_slow);
% fprintf('SH controller performance: %.2f  %.2f\n', perf_fast, perf_fast / perf_slow)
