classdef SelfCluteringController < ControllerBase
    %MULTICONTROLLER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        K
        shape
        gu
        gx
        objective
        box
        polyhedron
        mu
        Sigma
        net
        seq_length
        XTrain
    end
    
    methods        
        function obj = SelfCluteringController(objective_function, gx, gu, vertices)
            %MULTICONTROLLER Construct an instance of this class
            %   Detailed explanation goes here
            obj = obj@ControllerBase(size(gx, 1), size(gu, 1), gu);
            obj.objective = objective_function;
            obj.shape = length(objective_function);
            obj.gu = gu;
            obj.gx = gx;
            if ~isempty(vertices)
                obj = obj.generate_box_from_vertices(vertices);
                obj = obj.updateK();
            end
        end
        
        function obj = generate_box_from_vertices(obj, vertices)
            obj.box = vertices;%zeros(obj.shape, length(vertices));
%             for i=1:length(vertices)
%                 obj.box(:, i) = sat(vertices(:, i) ./ obj.objective', obj.gx);
%             end
        end
        
        function draw(obj, fig, color)
            if nargin < 3
                 color = rand(1, 3);
            end
            figure(fig);
            obj.polyhedron.plot('color', color, 'alpha', 0.4)
%             plot([obj.box(1, :), obj.box(1, 1)], [obj.box(2, :), obj.box(2, 1)], '-x', 'Color', color, 'MarkerSize', 12, 'LineWidth', 2)
            xlabel('e_1')
            ylabel('e_2')
        end
        
        function obj = updateK(obj)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            options = optimoptions('linprog','Display','none');
            C = [obj.box' zeros(size(obj.box, 2), obj.shape-size(obj.box, 1));zeros(size(obj.box, 2), obj.shape-size(obj.box, 1)) -obj.box'];
            A = [C;-C];
            b = repmat(obj.gu, size(A, 1) / size(obj.gu, 1), 1);
            obj.K = linprog(obj.objective,A, b, [], [], -inf*ones(size(obj.K)), inf*ones(size(obj.K)), ...
                            options)';
            if isempty(obj.K)
                return;
            end
            obj.K = reshape(obj.K, size(obj.box, 1), obj.shape / size(obj.box, 1))';
        end
        
        function answer = contains(obj, state)
            answer = obj.polyhedron.contains(state);
            %inpolygon(state(1), state(2), obj.box(1, :), obj.box(2, :));
        end
        
        function u = control_output(obj, state)
            u = obj.K * state;
            u = obj.sat(u);
        end
        
        function obj = train_on_data(obj, states)
            [cr, c2r] = obj.calculate_convergence_rate(states);
            obj.XTrain = c2r';
            for i=1:size(states, 2)
                states(:, i) = sat(states(:, i) ./ obj.objective(1:size(states, 1))', obj.gx);
            end
            obj.polyhedron = Polyhedron('V', states');
            try
                obj.polyhedron.computeHRep;
            catch
                return
            end
            ch =  convhull(states(1, :), states(2, :));
            vertices = obj.polyhedron.V;%states(:, ch(1:end-1));
%             figure; plot(states(1, :), states(2, :), '.', vertices(1, :), vertices(2, :), '-x')
            obj = obj.generate_box_from_vertices(vertices');
            obj = obj.updateK();
            obj.mu = mean(states');
            obj.Sigma = cov(states');
        end
        
        function prob = get_prob_for_state(obj, state)
            prob = mvnpdf(state, obj.mu, obj.Sigma);
        end
        
        function prob = get_prob_for_state_normalized(obj, state)
            prob = mvnpdf(state, obj.mu, obj.Sigma) / mvnpdf(state*0, obj.mu, obj.Sigma);
        end
        
        function obj = adapt(obj, state, next_state, u)
            if abs(u) < obj.gu && ~obj.contains(state) && obj.contains(next_state)
                obj.add_vertex_to_polyhedron(state);
                return
            end
            
            if ~obj.contains(state)
                return
            end
            
            if ~obj.contains(next_state)
                if abs(obj.K * state) < 0.9 * obj.gu && abs(obj.K * next_state) < 0.9 * obj.gu
                    obj.add_vertex_to_polyhedron(next_state);
                    disp('Expand and add next_state')
                else %if abs(obj.K * state) > obj.gu
                    disp('Contruct');
                end
                return
            else
                if abs(obj.K * state) < 0.9 * obj.gu
                    obj.expand(state);
                    disp('Expand')
                end
            end
        end
        
        function add_vertex_to_polyhedron(obj, state)
            obj.polyhedron = Polyhedron('V', [obj.polyhedron.V; state']);
            obj.polyhedron.computeHRep;
        end
        
        function expand(obj, state)
%             K = convhulln(obj.polyhedron.V);
            [V, D] = obj.get_vertices_and_distances(state);
%             figure(99);clf;obj.draw(99);hold on;plot(state(1), state(2), 'rx');
            [xs, index] = sort(D);n = 3;
            result      = V(index(1:n), :);
%             plot(result(:, 1), result(:, 2), 'bx')
            for i=1:n
                rho = V(index(i), :)' - state;
                u_rho = abs(obj.K * rho);
                u_left = obj.gu - abs(obj.K * V(index(i), :)');
                alpha_max = max(min(min(u_left ./ u_rho), 0.2), 0);
%                 quiver([V(index(i), 1), V(index(i), 1) + alpha_max * rho(1)],...
%                        [V(index(i), 2), V(index(i), 2) + alpha_max * rho(2)],...
%                        [rho(1), 0],...
%                        [rho(2), 0])
                   
                V(index(i), :) = V(index(i), :) + alpha_max * rho';
%                 plot(V(index(i), 1), V(index(i), 2), 'kx')
            end
            obj.polyhedron = Polyhedron('V', V);
        end
        
        function contruct(obj, state)
            
        end
        function obj = move_towards_state(obj, state)
            [V, D] = obj.get_vertices_and_distances(state);
            alpha = 0.1;
            for i=1:size(V, 1)
                normal_vector = V(i, :) - state';
                V(i, :) = V(i, :) - 1/(D(i)^3) * normal_vector;
            end
            obj.polyhedron = Polyhedron('V', V);
        end
        
        
        function move_away_from_state(obj, state)
            [V, D] = obj.get_vertices_and_distances(state);
            alpha = 0.1;
            for i=1:size(V, 1)
                normal_vector = V(i, :) - state';
                V(i, :) = V(i, :) + 1/(D(i)^2) * normal_vector;
            end
            obj.polyhedron = Polyhedron('V', V);
        end
        
        function [V, distances] = get_vertices_and_distances(obj, state)
            V = obj.polyhedron.V;
            distances = zeros(size(V, 1), 1);
            for i=1:size(V, 1)
                distances(i) = norm(V(i, :) - state');
            end
        end
        
        function traininfo = build_net_from_data(obj, data, layers, seq_length)
            obj.seq_length = seq_length;
            XTrain = zeros(2, seq_length, 1, length(data)-seq_length);
            YTrain = zeros(1, 1, 2, length(data)-seq_length);
            j = 1;
            for i=1:length(data)-seq_length
                if mean(data(:, i:i+seq_length-1)) < 0.1
                    continue;
                end
                XTrain(:, :, 1, j) = data(:, i:i+seq_length-1);
                YTrain(:, :, :, j) = data(:, i+seq_length);
                j = j + 1;
            end
            XTrain(:, :, :, j:end) = [];
            YTrain(:, :, :, j:end) = [];
            maxEpochs = 200;
            miniBatchSize = 27;

            options = trainingOptions('adam', ...
                'ExecutionEnvironment','cpu', ...
                'GradientThreshold',1, ...
                'MaxEpochs',maxEpochs, ...
                'MiniBatchSize',miniBatchSize, ...
                'SequenceLength','longest', ...
                'Shuffle','never', ...
                'Verbose',0, ...
                'Plots','none');
            [trained_net, traininfo] = trainNetwork(XTrain,YTrain,layers,options);
            obj.net = trained_net;
            disp('Done trainning network') 
        end
        
        function p_state = predict_next_state(obj, prev_states)
            p_state = obj.net.predict(prev_states);
        end
        
        function mD = closest_points(obj, states)
            [cr, c2r] = obj.calculate_convergence_rate(states);
            [mIdx,mD] = knnsearch(obj.XTrain,c2r','K',10,'Distance','minkowski','P',5);
        end
        
        function [cr, c2r] = calculate_convergence_rate(obj, states)
            cr = states(:, 2:end) - states(:, 1:end-1);
            c2r = cr(:, 2:end) - cr(:, 1:end-1);
        end
    end
end

