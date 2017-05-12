function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


test_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% List value pairs of c and sigma
rep_c = repelem(test_val, 8);
rep_sigma = repmat(test_val, 1, 8);
model_size = size(rep_c, 2);
p_error = zeros(1, model_size);

% Calculate p_error
for i=1:model_size
    calc_c = rep_c(i);
    calc_sigma = rep_sigma(i);
    
    % Train model
    model= svmTrain(X, y, calc_c, @(x1, x2) gaussianKernel(x1, x2, calc_sigma));
    
    % Make predictions, calc error
    predictions = svmPredict(model, Xval);
    p_error(i) = mean(double(predictions ~= yval));
    
end;

[~, min_idx] = min(p_error);

C = rep_c(min_idx);
sigma = rep_sigma(min_idx);

% =========================================================================

end
