%% training SVD model with addtional input
% \hat r_{i, j} = u + b_u + b_i + q_i^T \times (p_u +
%        |N(u)|^(0.5) \sum_{j \in N(u)} y_i)

clear; clc;

load('./Data/MovieLens/ml-latest-small/RM_train_test_split_1124.mat');
[numUser, numMovie] = size(RM_train);

% training params grid
% gamma: learning rate
% lambda: regularization
% f: hidden dimension
%params_grid = struct('lambda', [0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'f', [3,2, 4, 5], ...
%    'gamma', [0.001, 0.01, 0.1,  1, 10]);

params_grid = struct('lambda', [0.001], 'f', 3, 'gamma', [0.002]);
fp = [];
for i = 1 : length(params_grid.lambda)
    for j = 1 : length(params_grid.f)
        for k = 1 : length(params_grid.gamma)
            fp = [fp; params_grid.lambda(i), params_grid.f(j), params_grid.gamma(k)];
        end
    end
end

% set random seed
rng(0);

% active users and ratings, as well as crossvalidtion setup
CV = [1, 2, 3, 4, 5]
[KU, KI] = find(RM_train ~= 99);
numData = length(KU);
CV_prop = floor(numData/length(CV));

% randomize users and ratings
rand_idx = randperm(numData);
KU = KU(rand_idx);
KI = KI(rand_idx);

%% cross validation param tuning

% report the result
fid = fopen('SVD_additonal_inputs_113016.txt', 'w');


error = []
for fp_idx = 1:size(fp, 1)
    % current parameters
    lambda = fp(fp_idx, 1);
    f = fp(fp_idx, 2);
    gamma = fp(fp_idx, 3);
    
    % cross-validaiton
    cv_error = [];
    for c = 1:max(CV)
        % prepare cv data
        cv_idx = 1+(c-1)*CV_prop : c*CV_prop;
        cv_mask = false(1, numData);
        cv_mask(cv_idx) = true;
        train_mask = ~cv_mask;
        
        KU_cv = KU(cv_mask); KI_cv = KI(cv_mask);
        KU_tr = KU(train_mask); KI_tr = KI(train_mask);
        
        % matrix that only contains rated elements (set K)
        RM_tr = zeros(size(RM_train));
        mask_tr = false(size(RM_train));
        for j = 1 : length(KU_tr)
            mask_tr(KU_tr(j), KI_tr(j)) = true;
            RM_tr(KU_tr(j), KI_tr(j)) = RM_train(KU_tr(j), KI_tr(j));
        end
		
		mask_cv = false(size(RM_train));
		for j = 1 : length(KU_cv)
			mask_cv(KU_cv(j), KI_cv(j)) = true;
		end
        mask_N_u_cv = ~mask_cv;
		N_u_cv = sum(mask_cv, 2);
		
        % initialze the parameters
        % b_u, b_i, P, Q,
        % P: the column corresponds to p_u, Q: the columns corresponds to q_i
        % u: user, i: item
        b_u = rand(numUser, 1);
        b_i = rand(1, numMovie);
        P = rand(f, numUser);
        Q = rand(f, numMovie);
        Y = rand(f, numMovie); % implicit information
        
        
        % u can be calculated directly
        U = sum(RM_tr(:))/sum(mask_tr(:));
        % N(u) can be calculated directly as well
        N_u = sum(mask_tr, 2);
        mask_N_u = ~mask_tr;
        
        delta_J = 1000;
        J = [];
        J = [J, svd_additional_inputs_J(RM_tr, U, b_u, b_i, P, Q,Y, mask_tr, lambda)];
        iter = 1;
        
        while (delta_J > 1) && iter < 2000
            iter = iter + 1;
            % stochastic graident descent: loop through all the data
            for k = 1 : length(KU_tr)
                u = KU_tr(k);
                i = KI_tr(k);
                mask_N_u_u = repmat(mask_N_u(u,:), f, 1);
                
                % calculate error
                e_ui = RM_tr(u, i) - U - b_u(u) - b_i(i) - Q(:, i)' * ...
                    (P(:, u) + 1/sqrt(N_u(u)) * sum(mask_N_u_u.* Y, 2));
                
                % update
                b_u_u = b_u(u) - gamma * (lambda*b_u(u) - e_ui);
                b_i_i = b_i(i) - gamma * (lambda*b_i(i) - e_ui);
                P_u = P(:, u) - gamma * (lambda*P(:, u) - e_ui * Q(:, i));
                Q_i  =  Q(:, i) - gamma * (lambda*Q(:, i) - e_ui*P(:, u));
                % update Y(:, i)
                idx_N_u_i = find(mask_N_u(u, :));
                for kk = 1:length(idx_N_u_i)
                    idx_kk = idx_N_u_i(kk); % update Y(:, idx_kk)
                    Y(:, idx_kk) = Y(:, idx_kk) - gamma * (lambda * Y(:, idx_kk)  - e_ui/sqrt(N_u(u))*Q(:, i));
                end
                
                b_u(u) = b_u_u;
                b_i(i) = b_i_i;
                P(:, u) = P_u;
                Q(:, i) = Q_i;
            end
            
            cur_J = svd_additional_inputs_J(RM_tr, U, b_u, b_i, P, Q, Y, mask_tr, lambda);
            if isnan(cur_J)
                break
            end
            J = [J, cur_J];
            delta_J = abs(J(end) - J(end-1));
        end
        
        % calculate the average error for cross validation data
        accu_error = 0;
        for k = 1 : length(KU_cv)
			mask_N_u_cv_u = repmat(mask_N_u_cv(KU_cv(k),:), f, 1);
						
            r_hat = U + b_u(KU_cv(k)) + b_i(KI_cv(k)) + Q(:, KI_cv(k))' * ... 
					(P(:, KU_cv(k)) + 1/sqrt(N_u_cv(KU_cv(k))) * sum(mask_N_u_cv_u.*Y, 2));
            accu_error = accu_error + abs(r_hat - RM_train(KU_cv(k), KI_cv(k)));
        end
        cv_error = [cv_error, accu_error/length(KU_cv)];
        
    end     % cv
    error = [error, mean(cv_error)];
    fprintf(fid, '\n gamma: %f \t lambda: %f \t f: %f \t error: %f \n\n', gamma, lambda, f, error(end));
end % fp

fclose(fid)





