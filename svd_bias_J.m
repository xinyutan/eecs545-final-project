function [J] = svd_bias_J(RM_tr, u, b_u, b_i, P, Q, mask_tr, lambda)
% Calculate the objective function for svd_bias
% J = \sum_{(u,i) \in K} (r_{u,i} - u - b_u - b_i - p_u^T q_i)^2 + 
%	\lambda (|p_u|^2 + |q_i|^2 + b_u^2 + b_i^2)

numUser = size(RM_tr, 1);
numMovie = size(RM_tr, 2);

error_matrix = (RM_tr - u * mask_tr - repmat(b_u, 1, numMovie) .* mask_tr - repmat(b_i, numUser, 1) .* mask_tr - ...
		(P'*Q) .* mask_tr).^2;

J = sum(error_matrix(:)) + lambda * (sum(sum(P.^2)) + sum(sum(Q.^2)) + b_u'*b_u + b_i*b_i'); 

end
