function [ MTrain, MTest ] = split( M, rand_seed, frac )
%split(M, rand_seed, frac) split the rating matrix M into training and
%testing data
%   M: original data, M_{i, j}==99 means it's empty rating
%   rand_seed: rng(rand_seed), to make sure reproducibility
%   frac: fraction of testing data, e.g., 0.2

rng(rand_seed);
uniform_mask = rand(size(M,1), size(M, 2));

test_mask = (uniform_mask < frac);

MTest = M;
MTest(test_mask) = M(test_mask);
MTest(~test_mask) = 99;

MTrain = M;
MTrain(~test_mask) = M(~test_mask);
MTrain(test_mask) = 99;



end

