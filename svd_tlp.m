load('hw1_data.mat');

%% Manipulating X

%Finding non-zero indexes of X
nZeroIndex = find(X);

%Changing each entry x with log(x)+1
X(nZeroIndex) = log10(X(nZeroIndex))+1;

%% Collapsing X

Z = sum(X,3);

%% Computing SVD of Z
[U, S, V] = svd(Z, 0);

%Plotting singular values of Z
plot(diag(S), '*');
title('Singular values of Z')
%% rank-K approximations of Z

%rank-2
Z2 = U(:,1:2)*S(1:2,1:2)*V(:,1:2)';

%rank-10
Z10 = U(:,1:10)*S(1:10,1:10)*V(:,1:10)';

%rank-20
Z20 = U(:,1:20)*S(1:20,1:20)*V(:,1:20)';

%rank-50
Z50 = U(:,1:50)*S(1:50,1:50)*V(:,1:50)';

%rank-100
Z100 = U(:,1:100)*S(1:100,1:100)*V(:,1:100)';

%rank-300
Z300 = U(:,1:300)*S(1:300,1:300)*V(:,1:300)';

%% Manipulating Y

%Finding non-zero indexes of Y
nZeroIndex = find(Y);

%Changing each entry y with 1
Y(nZeroIndex) = 1;

%Vectorizing Y
Y = Y(:);

%% Vectorizing best rank-K approximations

%rank-2

Z2 = Z2(:);

%rank-10

Z10 = Z10(:);

%rank-20

Z20 = Z20(:);

%rank-50

Z50 = Z50(:);

%rank-100

Z100 = Z100(:);

%rank-300

Z300 = Z300(:);

%% plotting ROC curve

AUC_values = zeros(6, 1);

figure;

%rank-2
model = fitglm(Z2, Y,'Distribution','binomial','Link','logit');

scores = model.Fitted.Probability;

[AX, AY, T, AUC] = perfcurve(Y , scores, '1');

AUC_values(1) = AUC;

plot(AX,AY); hold on;

%rank-10
model = fitglm(Z10, Y,'Distribution','binomial','Link','logit');

scores = model.Fitted.Probability;

[AX, AY, T, AUC] = perfcurve(Y , scores, '1');

AUC_values(2) = AUC;

plot(AX,AY); hold on;

%rank-20
model = fitglm(Z20, Y,'Distribution','binomial','Link','logit');

scores = model.Fitted.Probability;

[AX, AY, T, AUC] = perfcurve(Y , scores, '1');

AUC_values(3) = AUC;

plot(AX,AY); hold on;

%rank-50
model = fitglm(Z50, Y,'Distribution','binomial','Link','logit');

scores = model.Fitted.Probability;

[AX, AY, T, AUC] = perfcurve(Y , scores, '1');

AUC_values(4) = AUC;

plot(AX,AY); hold on;

%rank-100
model = fitglm(Z100, Y,'Distribution','binomial','Link','logit');

scores = model.Fitted.Probability;

[AX, AY, T, AUC] = perfcurve(Y , scores, '1');

AUC_values(5) = AUC;

plot(AX,AY); hold on;

%rank-300
model = fitglm(Z300, Y,'Distribution','binomial','Link','logit');

scores = model.Fitted.Probability;

[AX, AY, T, AUC] = perfcurve(Y , scores, '1');

AUC_values(6) = AUC;

plot(AX,AY); hold on;

%labeling
lgd = legend(strcat('K=2, AUC=', num2str(AUC_values(1))), strcat('K=10, AUC=', num2str(AUC_values(2))), strcat('K=20, AUC=', num2str(AUC_values(3))), strcat('K=50, AUC=', num2str(AUC_values(4))), strcat('K=100, AUC=', num2str(AUC_values(5))), strcat('K=300, AUC=', num2str(AUC_values(6))));
title('ROC plots for Z');
lgd.Location = 'best';