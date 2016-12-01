function [J] = svd_additional_inputs_J(RM_tr, U, b_u, b_i, P, Q, Y, mask_tr, lambda)
% calculate the objective function for SVD_additional_inputs
% \hat r_{i, j} = u + b_u + b_i + q_i^T \times (p_u + 
%        |N(u)|^(0.5) \sum_{j \in N(u)} y_i)
% the columns of Y corresponds to items 

N_u = sum(mask_tr, 2);
numUser = size(RM_tr, 1);
numMovie = size(RM_tr, 2);

% calculate the matrix \sum_{j \in N(u)} y_i for all users, form a matrix
mask_unrated = ~mask_tr;

YU = zeros(size(P));

for u = 1 : length(b_u)
	mask_N_u_u = repmat(mask_unrated(u, :), size(P, 1), 1);
	YU(:, u) = 1/sqrt(N_u(u)) * sum(Y .* mask_N_u_u, 2);	
end

error_matrix = (RM_tr - U * mask_tr - repmat(b_u, 1, numMovie) .* mask_tr - ...
	repmat(b_i, numUser, 1) .* mask_tr - ((P + YU)' * Q) .* mask_tr).^2;

J = sum(error_matrix(:)) + lambda * (sum(sum(P.^2)) + sum(sum(Q.^2)) + b_u'*b_u + b_i*b_i' + sum(sum(Y.^2)));
  


end
