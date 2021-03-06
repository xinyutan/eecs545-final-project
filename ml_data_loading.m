%% load and split MovieLens data
clear; clc;
% load data

% four columns corresponds to: userId, movieId, rating, timestamp
data = csvread('./data/MovieLens/ml-latest-small/ratings.csv',1,0);

% initialize a matrix of which rows represent distinct users and columns
% distinct movies 
distinct_userId = unique(data(:,1)); % continuous
distinct_movieId = unique(data(:, 2)); % not continuous

rating_matrix = 99*ones(length(distinct_userId), length(distinct_movieId));

% fill up rating matrix, the unrated element will be 99

for idx = 1: length(data)
    c_userID = data(idx, 1);
    c_movieID = data(idx, 2);
    c_movieIdx = find(distinct_movieId == c_movieID);
    c_rating = data(idx, 3);
    
    rating_matrix(c_userID, c_movieIdx) = c_rating;
end

save('data/MovieLens/ml-latest-small/rating_matrix_1124.mat', 'rating_matrix');

[RM_train, RM_test] = split(rating_matrix, 0, 0.2);
save('data/MovieLens/ml-latest-small/RM_train_test_split_1124.mat', 'RM_train', 'RM_test');