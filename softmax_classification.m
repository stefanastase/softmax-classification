    clear; clc; close all
%% READING CSV
date = readtable('archive/fashion-mnist_train.csv');
[n, ~] = size(date);
X = date{:, 2:end};
X = X/256;
labels = date{:,1};
e = @(k) [zeros(k-1, 1); 1; zeros(10-k, 1)];
Y = zeros(10, n);
for i=1:n
    Y(:,i) = e(labels(i)+1);
end
Y = Y';

%% Gradient Descent using constant step
W = zeros(784,10);
b = ones(n, 10);
alfa = 0.1;
maxIter = 2000;
for j=1:10
    iter=0;
    grad = 1;
    while iter<maxIter && norm(grad) > 1e-2
        norm(grad)
        Z = X*W + b;
        O = softmax(Z);
        err = O-Y;
        grad = 0;
        for i=1:n
            grad = grad + X(i, :) * err(i, j);
        end
        grad = grad/n;
        W(:, j) = W(:, j) - alfa*grad';
%       Using bias
%       b(:, j) = b(:, j) - alfa * 1/n * sum(err(:, j));
        iter = iter+1;
    end
end

%% Stochastic Gradient Descent
W = zeros(784,10);
b = ones(n, 10);
alfa = 0.01;
maxIter = 2000;
for j=1:10
    iter=0;
    grad = 1;
    while iter<maxIter && norm(grad) > 1e-3
        norm(grad)
        Z = X*W;
        O = softmax(Z);
        err = O-Y;
        i = randi(60000,1);
        grad = X(i, :) * err(i, j);
        W(:, j) = W(:, j) - alfa*grad';
%       Using bias
%       b(:, j) = b(:, j) - alfa* err(i, j));
        iter = iter+1;
    end
end


%% Testing with data
date = readtable('archive/fashion-mnist_test.csv');
[m, ~] = size(date);
X_test = date{:, 2:end};
X_test = X_test/256;
labels_test = date{:,1};

%% Computing confussion matrix
mat_confuzie = zeros(10);
for i=1:m
    Z = X_test(i,:)*W + b(1,:);
    [~, label] = max(softmax(Z));
    mat_confuzie(labels_test(i) + 1, label) = mat_confuzie(labels_test(i) + 1, label) + 1;
end

%% Testing with image
path = 'imagini/Camasa/161.png';
imag = imread(path);
imag = im2double(imag)*255;
imag = reshape(imag',1,784);

X_test = imag/256;
[~, label] = max(X_test*W);
% Using bias
% [~, label] = max(X_test*W + b1);

fprintf('Obiectul face parte din clasa %d.', label-1);
