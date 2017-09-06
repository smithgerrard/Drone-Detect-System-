clear all;
close all;
clc;
addpath('./fly_7_19')
load('rotor_50.mat');
%load('approach_from35ft.mat');
%load('goaway_from5ft.mat');
% load('noise5.mat');
%one_rotor = Recording;
% Recording = feet5(:,1);
% Recording1 = noise5(:,1);
% windowSize = 50; 
% b = (1/windowSize)*ones(1,windowSize);
% a = 1;
% myRecording = Recording(1:13230)
% for xx = 1:size(myRecording,2)
%     y_train(:,xx) = filter(b,a,myRecording(:,xx));
% end
% myRecording = y_train;
Recording = rotor_50(:,152);
figure;
plot(linspace(0,0.3,length(Recording(1:13230))),Recording(1:13230));
xlabel('Time(s)')
ylabel('Magnitude')

%%%%%%%%%%%%%%%%
figure;
hold on;
set(gca,'Fontsize',12)

start_t = 0;
delta_t = 0.2;
end_t = 0.6;
Fs = 44100;
t_ = 1;
y_train = [];
% for t_ =start_t:delta_t:end_t

%myRecording=Recording(1+(j_-1)*Fs:Fs*j_);
%myRecording = Recording(round(t_*Fs)+1:round((t_+delta_t)*Fs));%+H(t_*Fs:(t_+delta_t)*Fs);
% myRecording1 = Recording1(round(t_*Fs)+1:round((t_+delta_t)*Fs));%+H(t_*Fs:(t_+delta_t)*Fs);
myRecording = Recording(t_:t_+13230);
windowSize = 50; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;
for xx = 1:size(myRecording,2)
    y_train(:,xx) = filter(b,a,myRecording(:,xx));
end
myRecording = y_train;

%myRecording = myRecording(10:length(myRecording));
myRecording = myRecording-mean(myRecording);
% myRecording1 = myRecording1-mean(myRecording1);

% xlabel('time [s]')
% ylabel('signal')
% title(strcat('recording',num2str(rectime),'s'))

L = length(myRecording);
% L1 = length(myRecording1);

int_time = 0.02;

%i_ = 5;%[10,25,50,100]
%Nsample = round(L/i_);
Nsample = round(int_time*Fs);
Noverlap = round(0.9*Nsample);
%Fs = 44100;
%NFFT = 2^nextpow2(L); % Next power of 2 from length of y
NFFT = 8192*8;%1024;4096;


%spectrogram(myRecording,Nsample,Noverlap,NFFT,Fs,'Yaxis')

[s,f,t,P] = spectrogram(myRecording,Nsample,Noverlap,NFFT,Fs,'Yaxis');

% [s1,f1,t1,P1] = spectrogram(myRecording1,Nsample,Noverlap,NFFT,Fs,'Yaxis');
% 
% hndl = surf(t+t_,f,(abs(P-P1)),'EdgeColor','none');   
%        axis xy; axis tight; colormap(jet); 
%       view(0,90);
%       xlabel('Time (s)');
%       ylabel('Frequency (Hz)');



surf(t,f,(abs(P)),'EdgeColor','none');   
       axis xy; axis tight; colormap(jet); 
      view(0,90);
      xlabel('Time (s)');
      ylabel('Frequency (Hz)');

hold on
ylim([0 200])
%xlim([1 2])

% end