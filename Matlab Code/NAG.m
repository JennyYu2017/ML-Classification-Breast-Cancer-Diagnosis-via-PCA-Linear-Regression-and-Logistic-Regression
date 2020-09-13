clear;
clc;
close all;

%% Load Data
load D_bc_tr; 
load D_bc_te; 


%% Normalize data
%training dataset
Xtrain = zeros(30,480);
for i = 1:30
    xi = D_bc_tr(i,:);
    mi = mean(xi);
    vi = sqrt(var(xi));
    Xtrain(i,:) = (xi - mi)/ vi;
end

%test dataset
Xtest = zeros(30, 89);
for i = 1:30
    xi = D_bc_te(i,:);
    mi = mean(xi);
    vi = sqrt(var(xi));
    Xtest(i,:) = (xi - mi)/ vi;
end
 
ytrain = D_bc_tr(31,:);
ytest = D_bc_te(31,:);
 
Datatrain = [Xtrain; ytrain];
Datatest = [Xtest; ytest];

%% Training using NAG Algorithm 
x0=zeros(31,1); %initial point
a=10^-9; % parameter \alpha, a small positive scalar, typically determined 
%    by trial and error.
K=150;% No.of iteration 
k = 1;
xk = x0;
yk = xk;
tk = 1;
t0=cputime;
f = f_wdbc(xk,Datatrain);
format long
format compact
while k < K
    gk = g_wdbc(yk, Datatrain);
    xk_new = yk - a*gk;
    tk1 = 0.5*(1+sqrt(1+4*tk^2));
    gak = (1-tk)/tk1;
    yk = (1-gak)*xk_new + gak*xk;
    fk = f_wdbc(xk_new,Datatrain);
    f = [f fk];
    k = k + 1;
    xk = xk_new;
    tk = tk1;
end
w=xk;
disp('number of iteration')
K

disp('Training cputime ')
cpu_training = cputime - t0

disp('Objective function at solution point:')
fs = f_wdbc(w,Datatrain)

%% Testing
t1=cputime;
Dt = [Xtest; ones(1,89)]; % Add 1 column for bias value

Result = w'*Dt;
TestLabel = zeros(1, length(Result));
FalsePos = 0;
FalseNeg = 0;
for ii = 1:length(Result)
    if Result(ii)< 0
        TestLabel(ii) = -1;  
        if ytest(ii) > 0 % False Negative
            FalseNeg = FalseNeg + 1;
        end
    else
        TestLabel(ii) = 1;
        if ytest(ii) < 0 % False Negative
            FalsePos = FalsePos + 1;
        end
    end
end
    


disp('number of false positive')
FalsePos

disp('number of false negative')
FalseNeg

disp('misclassification ratio:')
(FalsePos + FalseNeg)/89

disp('Testing cputime ')
cpt = cputime - t1

