clear all;
close all;
clc;
addpath('./fly_7_19')
%load('feet_5_25.mat');
% load('one.mat');
% load('two.mat');
% load('three.mat');
load('four.mat');
% load('rotor1.mat');
% load('rotor2.mat');
% load('rotor3.mat');
% load('rotor4.mat');
%load('fly_1s.mat');
%train_rotor = position(88201:end,:);
train_rotor = four(88201:132300,:);
% train_rotor = train_rotor(88201:end,:);
windowSize = 50; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;
for xx = 1:size(train_rotor,2)
    y_train(:,xx) = filter(b,a,train_rotor(:,xx));
end
train_rotor = y_train;

%myRecording = train_rotor(16:size(train_rotor,1),:);
myRecording = train_rotor;
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

stft_pca = stft.';
mu = mean(stft_pca);
[coeff,score, latent] = pca(stft_pca);
SelectNum = cumsum(latent)./sum(latent);
index = find(SelectNum >= 0.25);
pos = min(index);
XXhat = score(:,1:pos);
Xhat = score(:,1:pos) * coeff(:,1:pos)';
Xhat = bsxfun(@plus, Xhat, mu);
%stft_pca_new = stft_pca * coeff(:,1:pos);

data = [Xhat];
% data = data.';
[class,centroid]=kmeans(XXhat,1,'Replicates',20);
tbl = tabulate(class);
% for jj = 1:size(data,1)
%     dist(jj) = (data(jj,:) - centroid) * (data(jj,:) - centroid).';
% end
