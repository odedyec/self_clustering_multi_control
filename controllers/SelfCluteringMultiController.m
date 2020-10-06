classdef SelfCluteringMultiController < handle
    %MULTICONTROLLER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numOfControllers
        gu
        gx
        objectives
        controllers
        colors
        previous_state
        mu
        Sigma
    end
    
    methods        
        function obj = SelfCluteringMultiController(objective_functions, gx, gu, data)
            %MULTICONTROLLER Construct an instance of this class
            %   Detailed explanation goes here
            obj.numOfControllers = size(objective_functions, 1);
            obj.gu = gu;
            obj.gx = gx;
            obj.objectives = objective_functions;
            obj.controllers = cell(obj.numOfControllers, 1);
            obj.colors = [];
            if ~isempty(data)
                obj = obj.train_on_data(data);
            end
            obj.previous_state = {nan, nan, nan};  % [index, state]
        end
        
        function reset_all(obj)
            obj.previous_state = {nan, nan, nan};
        end
        
        function obj = train_on_data(obj, data, fig, xx, yy)
            show = false;
            if nargin > 2
                show = true;
                if isempty(obj.colors)
                    obj.colors = obj.create_random_colors();
                end
            end
            if show
                figure(fig);
                quiver(data(1, :), data(2, :), xx, yy, 'AutoScale', 'off', 'HandleVisibility','off');
                xlabel('e_1')
                ylabel('e_2')
            end
            obj.mu = mean(data');
            obj.Sigma = cov(data'); 
            
            

            %% %%%%%%%%%%%%%%%%%%%%%%%
            Probs = mvnpdf(data', obj.mu, obj.Sigma); l_min = min(Probs); l_max = max(Probs); dl = l_max - l_min;
            levels = l_min + dl * (1:obj.numOfControllers) / (obj.numOfControllers+1);
            k = 1;
            for i=1:obj.numOfControllers
                I = find(Probs > levels(i));
                if isempty(I)
                    obj.numOfControllers = obj.numOfControllers - 1;
%                     obj.controllers{i} = SelfCluteringController(obj.objectives(i, :), obj.gx, obj.gu, [obj.gx; -obj.gx]'); %single_cont.draw(1); single_cont.K
                    continue;
                end
                if show
                    hold on
                    plot(data(1, I), data(2, I), '.', 'MarkerSize', 14, 'Color', obj.colors(i, :))
%                     save_to_multi_images(fig, sprintf('../../scmc/images/level_%d',i));
                end
                obj.controllers{k} = SelfCluteringController(obj.objectives(i, :), obj.gx, obj.gu, []);
                obj.controllers{k} = obj.controllers{k}.train_on_data(data(:, I));
                if isempty(obj.controllers{k}.K)
                    obj.numOfControllers = obj.numOfControllers - 1;
                    obj.controllers(i)=[];
                else
                    k = k + 1;
                end
            end
            if show
                obj.drawAll(fig, false)
                x1 = -obj.gx(1):.5:obj.gx(1); x2 = -obj.gx(2):.5:obj.gx(2);
                [X1,X2] = meshgrid(x1,x2);
                F = mvnpdf([X1(:) X2(:)],obj.mu,obj.Sigma);
                F = reshape(F,length(x2),length(x1));
                figure(fig+1);surf(x1,x2,F);caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
                xlabel('x1'); ylabel('x2'); zlabel('Probability Density');
                hold on
                for i=1:obj.numOfControllers
                    surf([-obj.gx(1) -obj.gx(1) obj.gx(1) obj.gx(1), -obj.gx(1)], ...
                        [-obj.gx(2) obj.gx(2) obj.gx(2) -obj.gx(2) -obj.gx(2)],...
                        ones(5) * levels(i), 'FaceColor', obj.colors(i, :), 'FaceAlpha', 0.5)
                end
            end
        end
        
        function colors = create_random_colors(obj)
            colors = rand(obj.numOfControllers, 3);
        end
        
        function drawAll(obj, fig, should_clear)
            if isempty(obj.colors)
                 obj.colors = rand(obj.numOfControllers, 3);
            end
            if nargin < 3
                should_clear = true;
            end                
            figure(fig);
            if should_clear 
                clf;
            end
            hold on;
            my_legend = {};
            for i=1:obj.numOfControllers
                obj.controllers{i}.draw(fig, obj.colors(i, :));
                my_legend = add_to_list(my_legend, num2str(obj.controllers{i}.K));
            end
            xlabel('e_1')
            ylabel('e_2') 
            legend(my_legend)
        end
        
        function drawActual(obj, A, B, Fx, Fu, gx, gu, fig)
            if isempty(obj.colors)
                 obj.colors = rand(obj.numOfControllers, 3);
            end
            figure(fig);
            hold on;
            for i=obj.numOfControllers:-1:1
                Omega = draw_omega2(A, B, obj.controllers{i}.K, Fx, Fu, gx, gu, 0);
                Omega.plot('alpha', 0.25, 'color', obj.colors(i, :));
            end
        end
        
        function [K, i] = get_k_from_state(obj,state)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            i = obj.numOfControllers;
            K = obj.controllers{i}.K;
            while true
%                 if abs(K * state) < 1
                if obj.controllers{i}.contains(state)
                    return
                end
                i = i - 1;
                if i == 0
                    i = 1;
                    return;
                end
                K = obj.controllers{i}.K;
            end
        end
        
        function [u, i] = control(obj, state)
            [K, i] = obj.get_k_from_state(state);
            u = K * state;
%             u = sat(u, obj.gu);
%             if ~isnan(obj.previous_state{1})
%                 if norm(obj.previous_state{2})*3 > norm(state)
%                     obj.controllers{obj.previous_state{3}}.adapt(obj.previous_state{2}, state, obj.previous_state{1});
%                 end
%             end
%             obj.previous_state{1} = u;
%             obj.previous_state{2} = state;
%             obj.previous_state{3} = i;
        end
        
        function remove_controller(obj, index)
            if index > obj.numOfControllers
                return
            end
            obj.numOfControllers = obj.numOfControllers - 1;
            obj.controllers(index) = [];
        end
    end
end

