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


%% Minimize Logistic Function using BFGS Algorithm
%  Parameter
NumFeature = 30;
K = 4; % Number of Loop

% Objective function: f_wdbc
% Gradient: g_wdbc
% Set the initial point w0 and a convergence  tolerance eps
eps = 10^-7;
w = zeros(NumFeature+1,1);
%ones(1, NumFeature + 1)*10^-2; %Add 1 for bias

t0 = cputime;
k = 1;
xk = w;
Sk = eye(length(w));
fk = f_wdbc(xk,Datatrain);
gk = g_wdbc(xk,Datatrain);
dk = -Sk*gk;
ak = bt_lsearch(xk,dk,'f_wdbc','g_wdbc',Datatrain);
dtk = ak*dk;
xk_new = xk + dtk;
fk_new = f_wdbc(xk_new,Datatrain);
dfk = abs(fk - fk_new);
% er = max(dfk,norm(dtk));

while k < K
      gk_new = g_wdbc(xk_new,Datatrain);
      gmk = gk_new - gk;
      D = dtk'*gmk;
      if D <= 0
         Sk = I;
      else
         sg = Sk*gmk;
         sw0 = (1+(gmk'*sg)/D)/D;
         sw1 = dtk*dtk';
         sw2 = sg*dtk';
         Sk = Sk + sw0*sw1 - (sw2'+sw2)/D;
      end
      fk = fk_new;
      gk = gk_new;
      xk = xk_new;
      dk = -Sk*gk;
      ak = bt_lsearch(xk,dk,'f_wdbc','g_wdbc',Datatrain);
      dtk = ak*dk;
      xk_new = xk + dtk;
      fk_new = f_wdbc(xk_new,Datatrain);
      dfk = abs(fk - fk_new);
      % er = max(dfk,norm(dtk));
      k = k + 1;
end

w=xk_new;

disp('Number of iterations:')
K
disp('Objective function at the solution point:')
fs = f_wdbc(w,Datatrain)
format short

disp('training CPU time:')
cpu_training=cputime-t0


%%Testing: 
t1=cputime;
Dt = [Xtest; ones(1,89)];
Res = w'*Dt;
LabelOfTest = zeros(1, length(Res));
FalsePositive = 0;
FalseNegative = 0;
for ii = 1:length(Res)
    if Res(ii)< 0
        LabelOfTest(ii) = -1;  
        if ytest(ii) > 0 
            FalsePostive = FalsePositive + 1;
            
        end
    else
        LabelOfTest(ii) = 1;
        if ytest(ii) < 0 
            FalseNegative = FalseNegative + 1;
        end
    end
end    


disp('number of false positive')
FalsePositive
disp('number of false negative')
FalseNegative
disp('rate of misclassification in percentage')
(FalsePositive + FalseNegative)/89

disp('training CPU time:')
cpu_testing=cputime-t1





