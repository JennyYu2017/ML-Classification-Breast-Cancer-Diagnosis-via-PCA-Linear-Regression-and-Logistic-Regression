clear;
clc;
close all;

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


%% PCA Stage
q = 25;
i = 2;
t0=cputime;
m1= mean(X_tr_1,2);
A1= X_tr_1-m1*ones(1,p1);
[U1,~] = eigs(A1*A1',q);
% [u,s,v]=svd(A1);
% U1=u(:,1:q);


m2= mean(X_tr_2,2);
A2= X_tr_2-m2*ones(1,p2);
[U2,~] = eigs(A2*A2',q);
% [u,s,v]=svd(A2);
% U2=u(:,1:q);

m=[m1,m2];
U=[U1,U2];

%%Test Stage 
t0 = cputime;
Label_predict = zeros(1,89);
for jj = 1:89
    A_test = D_bc_te(1:30,jj); 
    e = zeros(1,2);

    for ii = 1:2
        At= A_test - m(:,ii);
        fj=U(:,(ii-1)*q+1:ii*q)'*At;
        Aj = U(:,(ii-1)*q+1:ii*q)*fj + m(:,ii);
        e(ii) = norm(A_test-Aj);   
    end
    [~,Tag] = min(e);
    if Tag==1
        Label_predict(1,jj) = 1;
    else
        Label_predict(1,jj) = -1;
    end
end
disp('No.of principal axis')
q
disp('Training cputime')
cpt = cputime - t0

%% Error Rate Display Stage
t1 = cputime;
Label = D_bc_te(31,:);
FalsePositive = 0;
FalseNegative = 0;

for i=1:89
    lt=Label(1,i);
    le=Label_predict(1,i);
    if lt~=le 
        if lt==1
            FalsePositive = FalsePositive+1;
        else
            FalseNegative = FalseNegative+1;
        end
    end
end  
E = (Label_predict~=Label);


disp('Number of Errors')
FalsePositive
FalseNegative
n_mis  = FalsePositive+FalseNegative
error_rate=n_mis/89

disp('Testing CPUtime:')
cpu_test=cputime-t1
