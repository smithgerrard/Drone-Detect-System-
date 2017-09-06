clear all;
close all;
clc;

load('rotor_50.mat')
% load('train.mat');
% load('test.mat');

train_rotor = [rotor_50(:,1:8),rotor_50(:,11:18),rotor_50(:,21:28),rotor_50(:,31:38),rotor_50(:,41:48),rotor_50(:,51:58),...
               rotor_50(:,61:68),rotor_50(:,71:78),rotor_50(:,81:88),rotor_50(:,91:98),rotor_50(:,101:108),rotor_50(:,111:118),...
               rotor_50(:,121:128),rotor_50(:,131:138),rotor_50(:,141:148),rotor_50(:,151:158),rotor_50(:,161:168),rotor_50(:,171:178),...
               rotor_50(:,181:188),rotor_50(:,191:198)];
test_rotor = [rotor_50(:,9:10),rotor_50(:,19:20),rotor_50(:,29:30),rotor_50(:,39:40),rotor_50(:,49:50),rotor_50(:,59:60),...
               rotor_50(:,69:70),rotor_50(:,79:80),rotor_50(:,89:90),rotor_50(:,99:100),rotor_50(:,109:110),rotor_50(:,119:120),...
               rotor_50(:,129:130),rotor_50(:,139:140),rotor_50(:,149:150),rotor_50(:,159:160),rotor_50(:,169:170),rotor_50(:,179:180),...
               rotor_50(:,189:190),rotor_50(:,199:200)];

windowSize = 50; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;
for xx = 1:size(train_rotor,2)
    y_train(:,xx) = filter(b,a,train_rotor(:,xx));
end
train_rotor = y_train;
for xx = 1:size(test_rotor,2)
    y_test(:,xx) = filter(b,a,test_rotor(:,xx));
end
test_rotor = y_test;

%load ('data');

% windowSize = 50; 
% b = (1/windowSize)*ones(1,windowSize);
% a = 1;
% for xx = 1:size(data,2)
%     y(:,xx) = filter(b,a,data(:,xx));
% end
% [Train1, Test1] = crossvalind('HoldOut', 20, 0.2);
% [Train2, Test2] = crossvalind('HoldOut', 20, 0.2);
% [Train3, Test3] = crossvalind('HoldOut', 20, 0.2);
% [Train4, Test4] = crossvalind('HoldOut', 20, 0.2);
% 
% data1 = y(:,1:20);
% train1 = data1(:,Train1);
% test1 = data1(:,Test1);
% 
% data2 = y(:,21:40);
% train2 = data2(:,Train2);
% test2 = data2(:,Test2);
% 
% data3 = y(:,41:60);
% train3 = data3(:,Train3);
% test3 = data3(:,Test3);
% 
% data4 = y(:,61:80);
% train4 = data4(:,Train4);
% test4 = data4(:,Test4);
% 
% train_rotor = [train1, train2, train3, train4];
% test_rotor = [test1, test2, test3, test4];

%% Spectrogram 

myRecording = train_rotor(16:size(train_rotor,1),:);
i = 50;
Nsample = round(size(myRecording,1)/i);
Noverlap = round(0.9*Nsample);
Fs = 44100;
NFFT = 4096;
num = size(myRecording,2);
stft = [];
for count = 1:num
%spectrogram(myRecording(:,1),Nsample,Noverlap,NFFT,Fs,'Yaxis')
    [s,f,t,P] = spectrogram(myRecording(:,count),hamming(Nsample),Noverlap,NFFT,Fs,'Yaxis');
    s = abs(s);
    [row, col] = size(s);
    s_temp = s(:);
    stft(:,count) = s_temp;
    s_temp = 0;
%    ss = reshape(s_temp,row,col);
end


% test
myRecording_test = test_rotor(16:size(test_rotor,1),:);
num_test = size(myRecording_test,2);
for count_test = 1:num_test
%spectrogram(myRecording(:,1),Nsample,Noverlap,NFFT,Fs,'Yaxis')
    [ss,ff,tt,PP] = spectrogram(myRecording_test(:,count_test),hamming(Nsample),Noverlap,NFFT,Fs,'Yaxis');
    ss = abs(ss);
    [row_test, col_test] = size(ss);
    ss_temp = ss(:);
    stft_test(:,count_test) = ss_temp;
    ss_temp = 0;
%    ss = reshape(s_temp,row,col);
end
    % hndl = surf(t,f,(abs(P))/1e-7,'EdgeColor','none');   
%       axis xy; axis tight;% colormap(jet); 
%       view(0,90);
%       xlabel('Time (s)');
%       ylabel('Frequency (Hz)');
% ylim([0 200])
%% PCA 
stft_pca = stft.';
mu = mean(stft_pca);
[coeff,score, latent] = pca(stft_pca);
SelectNum = cumsum(latent)./sum(latent);
index = find(SelectNum >= 0.90);
pos = min(index);
Xhat = score(:,1:pos) * coeff(:,1:pos)';
Xhat = bsxfun(@plus, Xhat, mu);
 
stft_pca_test = stft_test.';
mu_test = mean(stft_pca_test);
[coeff_test,score_test, latent_test] = pca(stft_pca_test);
SelectNum_test = cumsum(latent_test)./sum(latent_test);
index_test = find(SelectNum_test >= 0.90);
pos_test = min(index_test);
Xhat_test = score_test(:,1:pos_test) * coeff_test(:,1:pos_test)';
Xhat_test = bsxfun(@plus, Xhat_test, mu_test);


% mean_stft = mean(stft,2);
% mean_stft_test = mean(stft_test,2);
% sub_mean = [];
% sub_mean_test = [];
% for ii = 1:num
%     sub_mean(:,ii) = stft(:,ii) - mean_stft;    
% end
% 
% for iii = 1:num_test
%     sub_mean_test(:,iii) = stft_test(:,iii) - mean_stft_test;
% end
% 
% 
% cov = (sub_mean.'*sub_mean)/num;
% cov_test = (sub_mean_test.'*sub_mean_test)/num_test;
% 
% [V,D] = eig(cov);
% [VV,DD] = eig(cov_test);
% 
% [Y] = sort((diag(D)),'descend');
% [YY] = sort((diag(DD)),'descend');
% 
% V = sub_mean*V;
% VV = sub_mean_test*VV;
% 
% flip_V = fliplr(V);
% flip_VV = fliplr(VV);
% 
% %keep best 99% eigenvectors
% sum_Y=sum(Y);
% sum_YY=sum(YY);
% 
% c = 0;
% count = 0;
% for k=1:1:num
%     c=c+Y(k);
%     percent=c/sum_Y;
%     if percent>0.9
%         break;
%     end
%     if percent<0.9
%          count=count+1;
%     end
% end
% 
% cc = 0;
% count_test = 0;
% for kk=1:1:num_test
%     cc=cc+YY(kk);
%     percent_test=cc/sum_YY;
%     if percent_test>0.9
%         break;
%     end
%     if percent_test<0.9
%          count_test=count_test+1;
%     end
% end
% 
% train_eig_vect=V(:,1:count);
% test_eig_vect=VV(:,1:count_test);
% 
% %normalize eig_vect
%     for mm=1:1:count
%       train_eig_vect(:,mm)=train_eig_vect(:,mm)/norm(train_eig_vect(:,mm));
%     end
%     
%     for mmm=1:1:count_test
%       test_eig_vect(:,mmm)=test_eig_vect(:,mmm)/norm(test_eig_vect(:,mmm));
%     end
%    
% norm_train=sub_mean'*train_eig_vect;
% norm_test=sub_mean_test'*test_eig_vect;


%% Drone Detect

% find = [Xhat];
% 
% data = [Xhat_test];

% [class,centroid]=kmeans(find,1,'Replicates',20);
% tbl = tabulate(class);
% 
% for jj = 1:size(data,1)
%     dist(jj) = (data(jj,:) - centroid) * (data(jj,:) - centroid).';
% end



%% SVM 
train = stft.';
test = stft_test.';
% train = norm_train.';
% test = norm_test.';
train_label = [1*ones(40,1);2*ones(40,1);3*ones(40,1);4*ones(40,1)];
test_label = [1*ones(10,1);2*ones(10,1);3*ones(10,1);4*ones(10,1)];
%test_label = [4*ones(100,1)];
model = svmtrain(train_label, train, '-s 0 -t 0  -c 10  -g 0.01');
[predict_label, accuracy, dec_values] = svmpredict(test_label, test, model); % test the training data
%% confusion matrix
% t1=[ones(1,10), zeros(1,30)];
% t2=[zeros(1,10), ones(1,10), zeros(1,20)];
% t3=[zeros(1,20), ones(1,10), zeros(1,10)];
% t4=[zeros(1,30), ones(1,10)];

t1=[zeros(1,57)];
t2=[zeros(1,57)];;
t3=[zeros(1,57)];;
t4=[ones(1,57)];;

t=[t1;t2;t3;t4];
tt=zeros(4,57);
for l=1:1:length(predict_label)
    tt(predict_label(l),l)=1;
end
    plotconfusion(t,tt)
    
% t1=[ones(1,4),zeros(1,12)];
% t2=[zeros(1,4),ones(1,4),zeros(1,8)];
% t3=[zeros(1,8),ones(1,4),zeros(1,4)];
% t4=[zeros(1,12),ones(1,4)];
% t=[t1;t2;t3;t4];
% tt=zeros(4,16);
% for l=1:1:length(predict_label)
%     tt(predict_label(l),l)=1;
% end
%     plotconfusion(t,tt)

  