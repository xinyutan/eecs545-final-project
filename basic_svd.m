%-----------------------------------------------
% This script is to implement the basic version
% SVD:
% 	\hat r_{u,i} = q_i^T \times p_u
% objective function would be:
%	\min \sum (r_{u,i} - q_i^T * p_u)^2 + lambda*(p_u^2 + q_i^2)
%-----------------------------------------------
% doesn't work... Have a problem in converging....



clear; clc;

% load data
load('./data/MovieLens/ml-latest-small/RM_train_test_split_1124.mat');
[numUser, numMovie] = size(RM_train);

 
alpha_grid = [10, 100, 1]; % learning rate
lambda_grid = [0.01, 0.1, 1, 10, 50]; % regularization
f_grid = [2, 3, 4, 5]; % hidden dimension
CV = [1, 2, 3, 4, 5]; % cross_validation

% set randome seed
rng(0);

% active users and ratings
[KU, KI] = find(RM_train ~= 99);
numData = length(KU);
CV_prop = floor(numData/length(CV));

% randomize the user and item 

rand_idx = randperm(numData);
KU = KU(rand_idx);
KI = KI(rand_idx);

%% crossvalidation for parameter tuning

% record the result
fid = fopen('basic_SVD_112716.txt', 'w')

% objective function
J = @(RM_tr, P, Q, tr_mask, lambda) sum(sum((RM_tr - (P'*Q).*tr_mask).^2))/sum(tr_mask(:)) + lambda * (mean(mean(P.^2)) + mean(mean(Q.^2)) ); 

error = [];
for alpha_idx = 1:length(alpha_grid)
	for lambda_idx = 1:length(lambda_grid)
		for f_idx = 1:length(f_grid) 
			alpha = alpha_grid(alpha_idx);
			lambda = lambda_grid(lambda_idx);
			f = f_grid(f_idx);

					
			cv_error = [];
			for c = 1:max(CV)
				% prepare cross validation data
				cv_idx = 1+(c-1)*CV_prop : c*CV_prop;
				cv_mask = false(1, numData);
				cv_mask(cv_idx) = true;
				train_mask = ~cv_mask;
				KU_cv = KU(cv_mask); KI_cv = KI(cv_mask);
				KU_train = KU(train_mask); KI_train = KI(train_mask);
			
				%% training
				% initialize P and Q matrix
				% P: user, Q: movie
				P = rand(f, numUser) * 5;
				Q = rand(f, numMovie) * 5;
				
				RM_tr = zeros(size(RM_train));
				tr_mask = false(size(RM_train));
				for j = 1 : length(KU_train)
					tr_mask(KU_train(j), KI_train(j)) = true;
					RM_tr(KU_train(j), KI_train(j)) = RM_train(KU_train(j), KI_train(j));
				end
					
				% objective function
				delta_J = 100;
				J_array = [];
				J_array = [J_array, J(RM_tr, P, Q, tr_mask, lambda)];
				
				% train: optimize
				iteration = 0;
				while delta_J > 10^-3 && iteration < 1000
                    iteration = iteration + 1;
                    alpha_t = alpha/(1+iteration);
					% stochastic gradient descent, loop over all the data points
                    delta_q = [];
					for k = 1:length(KU_train)	
						u = KU_train(k); % user
						i = KI_train(k); % item
						e_ui = 	(RM_tr(u, i) - Q(:, i)'*P(:, u))/length(KU_train);
						
						q_i = Q(:, i) + alpha_t * (e_ui*P(:, u) - lambda*Q(:, i)/f);
						p_u = P(:, u) + alpha_t * (e_ui*Q(:, i) - lambda*P(:, u)/f);
						delta_q = [delta_q, sum((q_i-Q(:, i)).^2)];
						Q(:, i) = q_i;
						P(:, u) = p_u;
												
					end
					J_array = [J_array, J(RM_tr, P, Q, tr_mask, lambda)];
					delta_J = abs(J_array(end) - J_array(end-1));
				end
				
				% cross_validate
				accu_error = 0;
				for k = 1 : length(KU_cv)
					r_hat = Q(:, KI_cv(k))' * P(:, KU_cv(k));
					accu_error = accu_error + abs(r_hat - RM_train(KU_cv(k), KI_cv(k)));
				end
				cv_error = [cv_error, accu_error/length(KU_cv)];
				fprintf(fid, 'lamdba: %f, fold: %d, cv_error: cv_error(end)\n');
			end
			error = [error, mean(cv_error)];
			fprintf(fid, '\n alpha: %f, lambda: %f, f: %d \n error: %f \n\n', alpha, lambda, f, error(end));
		end
	end
end

fclose(fid)





