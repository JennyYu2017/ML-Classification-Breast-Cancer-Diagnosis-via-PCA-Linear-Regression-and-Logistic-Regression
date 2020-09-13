clear;
clc;
close all;
 
%%  Load Data & Normalize %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load D_bc_tr; %Training
load D_bc_te; %Testing
 
PLOT = 1;
%  Parameter
NumFeature = 30;
K = 75; % Number of Loop
 
%normalize test dataset
Xtrain = zeros(30,480);
for i = 1:30
    xi = D_bc_tr(i,:);
    mi = mean(xi);
    vi = sqrt(var(xi));
    Xtrain(i,:) = (xi - mi)/ vi;
end

%normalize test dataset
Xtest = zeros(30, 89);
for i = 1:30
    xi = D_bc_te(i,:);
    mi = mean(xi);
    vi = sqrt(var(xi));
    Xtest(i,:) = (xi - mi)/ vi;
end

 
ytrain = D_bc_tr(31,:);
ytest = D_bc_te(31,:);
 
Dtrain = [Xtrain; ytrain];
Dtest = [Xtest; ytest];
 
if PLOT == 1
%     figure, plot3(Xtrain(1,:), Xtrain(2,:), Xtrain(3,:), 'r*');
%     hold on;
%     plot3(Xtrain(1,:), Xtrain(2,:), Xtrain(5,:), 'bd');
%     xlabel('Feature 1');
%     ylabel('Feature 2');
%     zlabel('Value');
%     legend('Feature 3', 'Feature 4');
%     grid on;
    
    figure, plot(D_bc_tr(1,:),'b-');
    hold on;
    plot(D_bc_tr(3,:), 'r-');
    hold on;
    plot(Xtrain(1,:), 'k-');
    hold on;
    plot(Xtrain(3,:), 'y-');
    xlabel('Samples');
    ylabel('Value');
    legend('Feature 1 Before Norm', 'Feature 3 Before Norm','Feature 1 after Norm', 'Feature 3 after Norm');
end
 
%% Step E4.2 & E4.3
% Objective function: f_wdbc
% Gradient: g_wdbc
 
%% Strp E4.4: minimize f(w) using GD algorithm
% Step 1 : initial point w0 and a convergence  tolerance eps
eps = 10^-9;
w = zeros(NumFeature+1,1);
%ones(1, NumFeature + 1)*10^-2; %Add 1 for bias
% Initialize
f = zeros(1, K);
k = 1;
 
 
% Step 2 : Compute search direction d
gk = g_wdbc(w, Dtrain);
dk = - gk;
 
% Step 3 : Compute a positive scalar alpha_k using line_search
alpha_k = bt_lsearch(w,dk,'f_wdbc','g_wdbc', Dtrain);
 
 
 
 
%%%%%%%%%%%%%  Training %%%%%%%%%%%%%
 
while((norm(alpha_k*dk)  >= eps ) && (k<K))
    
    w = w + alpha_k*dk;
    % Step 2 : Compute search direction d
    gk = g_wdbc(w, Dtrain);
    dk = -gk;
    
    % Step 3 : Compute a positive scalar alpha_k using line_search
    alpha_k = bt_lsearch(w,dk,'f_wdbc', 'g_wdbc', Dtrain);
    
    % Save the value of f & see whether or not it's decreasing
    f(k) = f_wdbc(w, Dtrain);
    
    k = k + 1;
end
 
 
if PLOT == 1
    Dtr = [Xtrain; ones(1, 480)]; % Add 1 column for bias value
    AfterTr = w'*Dtr;
    Z = zeros(1,length(AfterTr));
    figure, plot(AfterTr, 'b*');
    hold on;
    plot(Z, 'r-');
    xlabel('Samples');
    ylabel('Value');
 
end
 
 
 
%%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%%%
 
Dt = [Xtest; ones(1,89)]; % Add 1 column for bias value
Result = w'*Dt;
TestLabel = zeros(1, length(Result));
FalsePos = 0;
FalseNeg = 0;
for ii = 1:length(Result)
    if Result(ii)< 0
        TestLabel(ii) = -1  
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
    
    
disp('Solution:')
w    
disp('number of false possitive')
FalsePos
disp('number of false negative in test data')
FalseNeg
disp('misclassification ratio:')
(FalsePos + FalseNeg)/89
 