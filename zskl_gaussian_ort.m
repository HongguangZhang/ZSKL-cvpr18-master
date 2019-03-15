% Zero Shot Learning Lab
function zskl_gaussian_ort(varargin)
addpath kernels;
addpath solver;

%% -------------------------------------------------------------------------
% Define the dataset 
dataset = 'AWA2';
load(sprintf('./data/%s/res101.mat', dataset)); %features: 2048xN
load(sprintf('./data/%s/att_splits.mat', dataset)); %labels: Nx1

%% -------------------------------------------------------------------------
% Data and attribute normalization
x_mean = mean(features(:, trainval_loc), 2);
for i = 1:size(features, 2)
    features(:,i) = (features(:,i) - x_mean);
    features(:,i) = features(:,i)./norm(features(:,i));
end

att_mean = mean(att, 2);
for i = 1:size(att,2)
    att(:,i) = att(:,i)-att_mean;
    att(:,i) = att(:,i)./norm(att(:,i));
end

% Train, val, test classes
total_cls = unique(labels);
test_unseen_cls = unique(labels(test_unseen_loc));
test_seen_cls = unique(labels(test_seen_loc));
train_cls = unique(labels(train_loc));
val_cls = unique(labels(val_loc));
trainval_cls = unique(labels(trainval_loc));

%% Randomly shuffle the train_loc
% Trainval - test. Please modify the split for cross-validation.
train_loc = trainval_loc(randperm(numel(trainval_loc)));
train_cls = trainval_cls;

%% Hyper Parameters
numIters = 12000; 
testIters = 400;
batchSize = 1;
checkpoint = 0;
LR = 8e-4; %Learning Rate
sigma = 0.8;
beta = 1;

%% Initializa the model
sc = 2/2048;
W = rand(2048, size(att, 1))*sc;

local_train_step = 0;
objective = 0;
avg_acc = 0;

state = 0;
stats = [];
best_zsl = 0;

%% Define the solver type
solver = @rmsprop;
solver_opts = struct('epsilon', 1e-2, 'rho', 0.99);

%% Define the loss function
funObj = @rbf_ort_loss;

%% Create directory and save checkpoints.
expDir = sprintf('./Gaussian-Ort-%s-data/BS%d-S%1.2f', dataset, batchSize, sigma);
if ~exist(expDir)
    mkdir(expDir);
    start = 1;
else
    Nf = numel(dir(expDir));
    if Nf>2
        load(fullfile(expDir, 'model.mat'), 'W', 'stats', 'state');
        checkpoint = numel(stats);
        start = checkpoint*testIters + 1;
        fprintf('Load Existing Model Sucessfully! \n');
    else
        start = 1;
    end
end

if start<numIters
  for i = start:numIters    
    phase = 'train';   
    index = rem((i-1)*batchSize:i*batchSize-1, numel(train_loc)) + 1;
    if index == 0; index = numel(train_loc); end

    x = features(:, train_loc(index));   
    y = labels(train_loc(index));    
    
    local_train_step = local_train_step + 1;
    [f, df, prediction] = funObj(x, W, att, y, sigma, train_cls, phase, beta);    
    accuracy = find(prediction==min(prediction))==find(train_cls==y);
    if numel(accuracy)~=1
        avg_acc = (avg_acc*(local_train_step-1) + accuracy(1))/local_train_step;
    else
        avg_acc = (avg_acc*(local_train_step-1) + accuracy)/local_train_step;
    end
    
    objective = (objective*(local_train_step-1) + f)/local_train_step;
    
    % SGD Optimizer
    [W, state] =solver(W, state, df, solver_opts, LR);               
    
    if rem(i,100)==0
        fprintf('Sigma:%0.2f ... Iter: %d/%d ... ', sigma, i, numIters);
        fprintf('Objective: %1.3f ... Top-1: %1.3f.\n', objective, avg_acc);
    end
    
    if rem(i, testIters)==0
        checkpoint = checkpoint + 1;
        loc = test_unseen_loc;
        cls = test_unseen_cls;
        x = features(:, loc);
        y = labels(loc);
        predict_label = zeros(size(y));
 
        A = W'*x;
        A_ = bsxfun(@minus, permute(repmat(A, [1 1 size(att(:,cls),2)]), [1 3 2]), att(:,cls));
        S1 = exp(-sum(A_.^2, 1)./(2*sigma^2));

        A = W*att(:,cls);
        A_ = bsxfun(@minus, A, permute(repmat(x, [1 1 size(att(:,cls), 2)]), [1 3 2]));
        S2 = exp(-sum(A_.^2, 1)./(2*sigma^2));
        score = permute((S1-1).^2 + (S2-1).^2, [3 2 1]);
        for j = 1:numel(loc)
            index = find(score(j,:)==min(score(j,:)));
            predict_label(j,1) = cls(index(1));
        end                         
        zsl = computeAcc(predict_label, y, cls);
        fprintf('\nZSL: Per-class accuracy: %1.2f  \n', zsl);
        
        loc = test_unseen_loc;
        cls = total_cls;
        x = features(:, loc);
        y = labels(loc);
        score = zeros(numel(loc), numel(cls));
        prediction = zeros(numel(loc), numel(cls));
        predict_label = zeros(size(y));
 
        A = W'*x;
        A_ = bsxfun(@minus, permute(repmat(A, [1 1 size(att,2)]), [1 3 2]), att);
        S1 = exp(-sum(A_.^2, 1)./(2*sigma^2));

        A = W*att;
        A_ = bsxfun(@minus, A, permute(repmat(x, [1 1 size(att, 2)]), [1 3 2]));
        S2 = exp(-sum(A_.^2, 1)./(2*sigma^2));
        score = permute((S1-1).^2 + (S2-1).^2, [3 2 1]);
        for j = 1:numel(loc)
            index = find(score(j,:)==min(score(j,:)));
            predict_label(j,1) = cls(index(1));
        end                         
        gzsl_u = computeAcc(predict_label, y, test_unseen_cls);
        fprintf('GZSL Unseen: Per-class accuracy: %1.2f  \n', gzsl_u);
        
        loc = test_seen_loc;
        cls = total_cls;
        x = features(:, loc);
        y = labels(loc);
        predict_label = zeros(size(y));
 
        A = W'*x;
        A_ = bsxfun(@minus, permute(repmat(A, [1 1 size(att,2)]), [1 3 2]), att(:,cls));
        S1 = exp(-sum(A_.^2, 1)./(2*sigma^2));

        A = W*att;
        A_ = bsxfun(@minus, A, permute(repmat(x, [1 1 size(att, 2)]), [1 3 2]));
        S2 = exp(-sum(A_.^2, 1)./(2*sigma^2));
        score = permute((S1-1).^2 + (S2-1).^2, [3 2 1]);
        for j = 1:numel(loc)
            index = find(score(j,:)==min(score(j,:)));
            predict_label(j,1) = cls(index(1));
        end
        
        gzsl_s = computeAcc(predict_label, y, test_seen_cls);
        fprintf('GZSL Seen: Per-class accuracy: %1.2f  \n', gzsl_s);
        
        gzsl_H = 2*gzsl_u*gzsl_s/(gzsl_u + gzsl_s);
        fprintf('GZSL H: Per-class accuracy: %1.2f  \n\n', gzsl_H);       
        
        if zsl > best_zsl
            best_zsl = zsl;
            stats(checkpoint).train = avg_acc;
            stats(checkpoint).zsl = zsl;
            stats(checkpoint).gzsl_u = gzsl_u;
            stats(checkpoint).gzsl_s = gzsl_s;
            stats(checkpoint).gzsl_H = gzsl_H;
            stats(checkpoint).trn_objective = objective;
            save(fullfile(expDir, sprintf('model.mat')), 'W', 'state', 'stats', 'sigma');
        end
        objective = 0;
        avg_acc = 0;
        local_train_step = 0;
    end
  end
end

function acc_per_class = computeAcc(predict_label, true_label, classes) 
    nclass = length(classes);
    acc_per_class = zeros(nclass, 1);
    for i=1:nclass
        idx = find(true_label==classes(i));
        acc_per_class(i) = sum(true_label(idx) == predict_label(idx)) / length(idx);
    end
    acc_per_class = mean(acc_per_class);
        
