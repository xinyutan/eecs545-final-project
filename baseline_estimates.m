%% Model 1 - Baseline estimates
% implements the 2.1 section in paper 'Factorization Meets the Neighborhood'
% $\hat r_{ui} = u + b_u + b_i $
% optimization problem:
%       \min_{b} \sum_{(u,i)\in \Kappa}(r_{ui}-u-bi_i-bi_u)^2 + \lambda_1(\sum_u b_u^2 + \sum_i b_i^2)

% load data
load('./data/MovieLens/ml-latest-small/RM_train_test_split_1124.mat');

alpha = 10; % learning rate
lambda_grid = [0.1, 1, 5];
CV = [1, 2, 3, 4, 5];

% set random seed
rng(0);


% active user and ratings
[KU, KI]  = find(RM_train~=99);
n = length(KU); % number of active ratings
CV_prop = floor(n/length(CV));

% randomize the rated items/user pair
rand_idx = randperm(n);
KU = KU(rand_idx); KI = KI(rand_idx);

error = [];
for l = 1:length(lambda_grid)
    lambda = lambda_grid(l);
    cv_error = [];
    for c = 1:length(CV)
        %% prepare the training and cross validation data
        CV_idx = 1 + (CV(c)-1)*CV_prop : CV(c)*CV_prop;
        CV_mask = false(1, n);
        CV_mask(CV_idx) = true;
        train_mask = ~CV_mask;
        
        KU_cv = KU(CV_mask); KI_cv = KI(CV_mask);
        KU_train = KU(train_mask); KI_train = KI(train_mask);
        
        %% training
        % reconstruct the training matrix using the current training data
        % make sure that that all empty elements are zeros, so that they don't
        % affect later computation
        
        RM_tr = zeros(size(RM_train));
        tr_mask = false(size(RM_train)); % the mask for the rated user/item pair
        for idx = 1:length(KU_train)
            tr_mask(KU_train(idx), KI_train(idx)) = true;
            RM_tr(KU_train(idx), KI_train(idx)) = RM_train(KU_train(idx), KI_train(idx));
        end
        
        % TODO: need to check if there are all-zeros row or column
        
        % user base line - need to learn
        bu = rand(size(RM_train, 1), 1)*5; % user - row
        bi = rand(1, size(RM_train, 2))*5; % item - column
        % reconstruct the matrix for BU and BI to simplify updates
        BU = repmat(bu, 1, length(bi));
        BU(~tr_mask) = 0;
        BI = repmat(bi, length(bu), 1);
        BI(~tr_mask) = 0;
        
        % overall average rating
        u = sum(RM_tr(:))/sum(tr_mask(:));
        U = u*ones(size(RM_Tr));
        U(~tr_mask) = 0;
        
        n_tr = sum(tr_mask(:));
        tic
        delta_J = 100; % difference between objective function
        J = [];
        J = [J, sum(sum((RM_tr-U-BU-BI).*(RM_tr-U-BU-BI)))/n_tr + lambda * (sum(bu.^2) + sum(bi.^2))];
        % TODO: repeat the follwoing steps until convergence
        iteration = 0;
        tic
        while delta_J > 10^(-3) && iteration < 1000
            iteration = iteration + 1;
            grad_bu = - 2/n_tr * sum(RM_tr-U-BU-BI, 2) + 2*lambda*bu/length(bu);
            grad_bi = -2/n_tr * sum(RM_tr-U-BU-BI, 1) + 2*lambda*bi/length(bi);
            
            %update
            bu = bu - alpha*grad_bu;
            bi = bi - alpha*grad_bi;

            BU = repmat(bu, 1, length(bi));
            BU(~tr_mask) = 0;
            BI = repmat(bi, length(bu), 1);
            BI(~tr_mask) = 0;            
            
            % calculate the J 
            J = [J, sum(sum((RM_tr-U-BU-BI).*(RM_tr-U-BU-BI)))/n_tr + lambda * (sum(bu.^2) + sum(bi.^2))];
            delta_J = abs(J(end)- J(end-1));
        end
        toc
        
        %% cross-validate and report the error
        accu_error = 0
        for idx = 1 : length(KU_cv)
            r_hat = u + bu(KU_cv(idx)) + bi(KI_cv(idx));
            accu_error = accu_error + abs(r_hat - RM_train(KU_cv(idx), KI_cv(idx)));
        end
        cv_error = [accu_error/length(KU_cv), cv_error];
        fprintf('lambda: %f, fold: %d, cv_error: %f \n', lambda, c, cv_error(end))
    end
    error = [mean(cv_error), error];  
end
