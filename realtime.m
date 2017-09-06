clear all;
close all;
clc;

load('centroid_pca_1s');
load('model');
[y,f] = audioread('beep-06.wav');
robj = audiorecorder(44100,24,1);  %Sampling frequency, bits, channel 
recordblocking(robj,3);            %Collcting 3s data
rdata = getaudiodata(robj);        %Store data
plot(rdata);                       %Plot data  
axis([1,44100,-0.005,0.0005]);    %Set axis  
drawnow                            %Refresh  
n = 100;                            %Set Refresh times   
m = 0.1;                           %Set time  
i = 50;
olddata1 = rdata;
Fs = 44100;
NFFT = 4096;
stft = [];
Nsample = round(44100/i);
Noverlap = round(0.9*Nsample);
test_label =4;


while n>1  
   recordblocking(robj,m);  
   rlen = length(rdata);                      
   olddata = rdata(floor(rlen*m):rlen,1);     %Old Dtat 
   rdata = [olddata ; getaudiodata(robj)];    %Dispaly data = old + new  
   rdata1 = [olddata1 ; getaudiodata(robj)];
   olddata1 = rdata1;
   % rdata = [getaudiodata(robj)];
    
   len = length(rdata1);
   myRecording = rdata1(len-44099:end);     %[3s,132300], [1s,44100] 
    
   [s] = spectrogram(myRecording,hamming(Nsample),Noverlap,NFFT,Fs,'Yaxis');
   s = abs(s);
   [row, col] = size(s);
   stft = s(:).';
   mu = mean(stft);
   [coeff,score, latent] = pca(stft);
   SelectNum = cumsum(latent)./sum(latent);
   index = find(SelectNum >= 0.95);
   pos = min(index);
   Xhat = score(:,1:pos) * coeff(:,1:pos)';
   Xhat = bsxfun(@plus, Xhat, mu);
   dist = (stft - centroid) * (stft - centroid).';
   display(dist);
   if dist > 20
        ;
   else
%       sound(y,f);
        [predict_label,~,~] = svmpredict(test_label, stft, model);
        fprintf(strcat(num2str(predict_label),'rotor','drone'));
   end
        stft =[];
    %    ss = reshape(s_temp,row,col);
  
    plot(rdata);  
    axis([1,44100,-0.0005,0.0005]);  
    drawnow  
    n = n-1;  
    
end  