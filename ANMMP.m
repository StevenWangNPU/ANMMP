function W = ANMMP(X_train, Y_train, k_inner, k_outer, feature_num)
% X_train: training data, each row is one data
% Y_train: corresponding labels of training data, a column vector
% k_inner: k_inner nearest neighbors in the same class 
% k_outer: k_outer nearest neighbors in different classes

if nargin < 5 
    error('Not enough input arguments.');
end;

if feature_num > size(X_train, 2)
    error('feature_num is too large.');
end;

[Sb_n, Sw_n] = calculate_neighbor_L(X_train,Y_train, k_inner, k_outer);  %计算k-nearest的Sw,sb
perr = 1e-6;
W = opt_TRR(Sb_n, Sw_n, feature_num, perr); %优化方法，计算W





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate neightborhood Sb and Sw
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Sb, Sw] = calculate_neighbor_L(X_train, Y_train, k_inner, k_outer)
% X_train: training data, each row is one data
% Y_train: corresponding labels of training data, a column vector
% k_inner: k_inner nearest neighbors in the same class 
% k_outer: k_outer nearest neighbors in different classes



[n_t, ~] = size(X_train);  %n_t样本的个数

% calculate the scatter matrices
WW_w = spalloc(n_t,n_t,20*n_t);  % spalloc产生一个n_t*n_t,有20*n_t个非零元素的稀疏矩阵，目的是减少存储空间和加快运算速度
WW_b = spalloc(n_t,n_t,20*n_t);

for j = 1:n_t
    % inner 找同类中距离最近的样本
    ind = find(Y_train == Y_train(j));    %ind中存储和j样本同类样本的label
    if length(ind) < k_inner+1            %k_inner表示在每个类别中
        error('k_inner is too large.');
    end;
    data = X_train(ind,:);
    n = size(data,1);                     %计算这个样本中的个数
    rep_j = repmat(X_train(j,:),n,1);     %将j样本复制成和同类样本相同个数的矩阵
    distance = rep_j - data;              %计算j样本和每个同类样本的距离，每个特征的差值
    distance = sum(distance.*distance,2); %海明距离
    [dumb,I] = sort(distance);            %距离排序 I中存储从最近到最远的样本的label
    WW_w(j,ind(I(2:k_inner+1))) = 1;      %I中存储的第一个元素是j样本本身，所以从第二个元素开始算起，在WW_w这个稀疏矩阵中，如果有最近的则将元素置为1
    % outer
    ind = find(Y_train ~= Y_train(j));
    if length(ind) < k_outer
        error('k_outer is too large.');
    end;
    data = X_train(ind,:);
    n = size(data,1);
    rep_j = repmat(X_train(j,:),n,1);
    distance = rep_j - data;
    distance = sum(distance.*distance,2);
    [dumb,I] = sort(distance);
    WW_b(j,ind(I(1:k_outer))) = 1;
end;

%A_w = WW_w.*WW_w';
%A_b = WW_b.*WW_b';

A_w = spones(WW_w+WW_w');
A_b = spones(WW_b+WW_b');

AA_w = (A_w+A_w')/2;
D_w = spdiags(sum(AA_w,2),0,n_t,n_t);
L_w = D_w - AA_w;

AA_b = (A_b+A_b')/2;
D_b = spdiags(sum(AA_b,2),0,n_t,n_t);
L_b = D_b - AA_b;

L_w = (L_w + L_w')/2;
L_b = (L_b + L_b')/2;
Sw = X_train'*L_w*X_train;
Sb = X_train'*L_b*X_train;

Sw = (Sw + Sw')/2;
Sb = (Sb + Sb')/2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% solving the constrained trace ratio optimization problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, lamda]  = opt_TRR(Sb, Sw, feature_num, p_err)
% Sb, Sw: scatter matrices
% feature_num: reduced dimensionality
% p_err: the error to optimum


[evec_sw eval_sw] = eig(Sw);
eval_sw = abs(diag(eval_sw));
nzero_sw = length(find(eval_sw<=1e-6));
if feature_num <= nzero_sw
    [dumb, iEvals] = sort(eval_sw);
    Z = evec_sw(:,iEvals(1:nzero_sw));
    [evec_sb eval_sb] = eig(Z'*Sb*Z);
    [dumb, iEvals] = sort(diag(eval_sb), 'descend');
    W = Z * evec_sb(:,iEvals(1:feature_num));
else
    [evec_sb eval_sb] = eig(Sb);
    eval_sb = sort(diag(eval_sb), 'descend');
    max_numerator = sum(eval_sb(1:feature_num));
    [evec_sw eval_sw] = eig(Sw);
    eval_sw = sort(diag(eval_sw));
    min_denominator = sum(abs(eval_sw(1:feature_num)));
    lamda_sup = max_numerator/min_denominator;
    lamda_inf = trace(Sb)/trace(Sw);
    interval = lamda_sup - lamda_inf;
    lamda = (lamda_inf+lamda_sup)/2;
    while interval > p_err
        [evec eval] = eig(Sb - lamda*Sw);
        [eval, index] = sort(diag(eval),'descend');
        sum_eval = sum(eval(1:feature_num));
        if sum_eval > 0
            lamda_inf = lamda;
        else
            lamda_sup = lamda;
        end;
        interval = lamda_sup - lamda_inf;
        lamda = (lamda_inf+lamda_sup)/2;
    end;
    W = evec(:,index(1:feature_num));
end;


