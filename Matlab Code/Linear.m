clear
clc
close all

%% Load data 
load D_bc_tr; 
load D_bc_te; 


    
%%Identify different group
p1=0;
p2=0;

for i=1:480
    ti=D_bc_tr(31,i);
    if ti==1
        p1=p1+1;
        X_tr_1(:,p1)=D_bc_tr(1:30,p1+p2);
    else
        p2=p2+1;
        X_tr_2(:,p2)=D_bc_tr(1:30,p1+p2);
    end 
end


p3=0;
p4=0;

for i=1:89
    ti=D_bc_tr(31,i);
    if ti==1
        p3=p3+1;
        X_te_1(:,p3)=D_bc_tr(1:30,p3+p4);
    else
        p4=p4+1;
        X_te_2(:,p4)=D_bc_tr(1:30,p3+p4);
    end 
end

X_tr=[X_tr_1,X_tr_2];
X_te=[X_te_1,X_te_2];
y = [ones(p1,1);-ones(p2,1)];

%% Plot different group by features
    % Feature 1: 
    figure,plot(X_tr_1(1,:),'r');
    hold on
    plot(X_tr_2(1,:),'b-');
    hold on 
    legend('Malignant','Benign');
    xlabel('samples');
    ylabel('Feature 1');
    
    %Feature 6:
    figure,plot(X_tr_1(6,:),'r-');
    hold on;
    plot(X_tr_2(6,:),'b-')
    hold on;
    legend('Malignant','Benign');
    xlabel('samples');
    ylabel('Feature 6');
    
     %Feature 18:
    figure,plot(X_tr_1(18,:),'r-');
    hold on;
    plot(X_tr_2(18,:),'b-')
    hold on;
    legend('Malignant','Benign');
    xlabel('samples');
    ylabel('Feature 18');
    
    
     %Feature 25:
    figure,plot(X_tr_1(25,:),'r-');
    hold on;
    plot(X_tr_2(25,:),'b-')
    hold on;
    legend('Malignant','Benign');
    xlabel('samples');
    ylabel('Feature 25');
    
       %Feature 30:
    figure,plot(X_tr_1(30,:),'r-');
    hold on;
    plot(X_tr_2(30,:),'b-')
    hold on;
    legend('Malignant','Benign');
    xlabel('samples');
    ylabel('Feature 30');
    
%%Linear model
t0=cputime;
Xh = [X_tr', ones(480,1)];
Xy = Xh'*y;
w = (Xh'*Xh)\Xy;
w1=w(1:30)
b1=w(31)

disp('Training cputime')
cpt = cputime - t0

%% Testing
t1=cputime;
Y_predicted=w1'*X_te+b1;
Falsepositive = 0;
Falsenegative = 0;

for i=1:p3
    if Y_predicted(:,i)<0
       Falsepositive=Falsepositive+1;
    end
end

p=p3+1;
for i=p:89
    if Y_predicted(:,i)>0
        Falsenegative=Falsenegative+1
    end
end


% Showing testing error*
disp('Number of miss class')
Falsepositive
Falsenegative
n_mis = Falsepositive+Falsenegative

disp('Error rate')
error_rate=n_mis/89

disp('Testing cputime')
cpt = cputime - t1