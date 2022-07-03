% project!!
%part a)
% a)

t = 0:1/1000:10; %will define t variable starting from zero,to a possitive value like step
xt =exp(-0.1.*t).*(cos(0.5*pi*t) + cos(50*pi*t));%no need for step cause for evry t<0 x(t)=0

figure(1);
      plot(t,xt);
      title('exp(-0.1*t)*(cos(0.5*pi*t) + cos(50*pi*t))')

      xlabel('t [sec]')
      ylabel('x(t)')
      grid on 


% b)%from the laplace table will see that 
% L{x(t)} = (s+0.1)/((s+0.1)^2+0.25*pi^2) + (s+0.1)/((s+0.1)^2+2500*pi^2)

s = tf('s'); %we need the actual filtering to be between transfer functions.

            
Lxt = (s+0.1)/((s+0.1)^2+0.25*pi^2) + (s+0.1)/((s+0.1)^2+2500*pi^2);

% d)
R = 3*10^3; 
C = 21.22*10^(-6); 
Hs = tf([1],[R*C 1]);


% e)
figure(2)
      bode(Hs);

figure(3);
      pzmap(Hs);
      grid on
      title('transmition function pole and zero map of out filter');



% f)
w_0 = logspace(-1,3); %Generate logarithmically spaced vector 
freq_H = freqs([1],[R*C 1],w_0); %Frequency response of analog filters 
magnitude_1 = abs(freq_H);
phase_1 = angle(freq_H);
phase_1 = phase_1*180/pi; %convert to degrees 

figure(4);
      loglog(w_0,magnitude_1); % Log-log scale plot
      title(' Frequency Response')
      xlabel('w [rad/sec]')
      ylabel('magnitude [dB]')

% filtering 
y_s = Lxt*Hs; % actual filtering 

figure(5);
subplot(2,1,2)
  impulse(y_s,t); %finds x_t for time domain
  title('x(t) After Noise filtration');
  ylabel('x(t)')
  xlabel('t[sec]');
  grid on;
  hold on
 
 subplot(2,1,1)
      plot(t,xt);
      title('x(t) before Noise filtration')
      ylabel('x(t)')
      xlabel('t[sec]');
       xlim([0 10]);
       grid on
  
  
  % part a_ e)

  Fs_e = 100; % the sampling frequency
T_s_e = 1/Fs_e; % the sampling time

ts_e = 0:T_s_e:10; % the time at the sample points
% x(t)
x = @(t) exp(-0.1.*t).*(cos(0.5*pi*t) + cos(50*pi*t));
% the discrete sequence
x_n = x(ts_e);

  figure(6);  
      plot(t,x(t));
      hold on
      % plot sampled as well 
 
      stem(ts_e, x_n);

      title('x_n  sampled signal the rate of f_s=100 [Hz]  ');
      xlabel('t [sec])');
      ylabel('x(t) )');

      legend('x(t)', 'x_n')
      grid on
  
  
  
  
  %% part a_ g
  

  omega=-4*pi:0.001:4*pi;
  
  x_omega=(exp(1i.*omega).*(exp(1i.*omega)-exp(-0.1.*T_s_e).*cos(0.5.*pi.*T_s_e)))./(exp(2.*1i.*omega)-(2.*exp(-0.1.*T_s_e).*cos(0.5.*pi.*T_s_e)).*exp(1i.*omega)+exp(-0.2.*T_s_e))+(exp(1i.*omega).*(exp(1i.*omega)-exp(-0.1.*T_s_e).*cos(50.*pi.*T_s_e)))./(exp(2.*1i.*omega)-(2.*exp(-0.1.*T_s_e).*cos(50.*pi.*T_s_e)).*exp(1i*omega)+exp(-0.2.*T_s_e));
  
  figure(7);
  subplot(2,1,1)
       plot(omega,abs(x_omega))
      xlabel('\Omega [rad/sample]')
      ylabel('|X(\Omega)|')
      title('DTFT of amplitude  x[n] at smapling rate  f_s=100 [Hz]');
  
       hold on
  
  subplot(2,1,2)
  
      plot(omega,angle(x_omega))
      title('DTFT of phase  x[n] at smapling rate  f_s=100 [Hz]');
      xlabel('\Omega [rad/sample]')
      ylabel('<X(\Omega)')
  
  
  %part a: h)
  
  Fs_h = 25; % the sampling frequency
T_s_h = 1/Fs_h; % the sampling time

ts_h = 0:T_s_h:10; % the time at the sample points
%sampling in other rate

x_n_h = x(ts_h);

  figure(8);  
      plot(t,x(t));
      hold on
      % plot sampled as well 
 
      stem(ts_h, x_n_h);

      title('x_n  sampled signal the rate of f_s=25 [Hz]  ');
      xlabel('t [sec])');
      ylabel('x(t) )');

       legend('x(t)', 'x_n')
      grid on
  
  
  
  
  %% part b 
  
  
  
%   
%   
%   figure;
%  
% % Signal and delay values.
% N = 100;
% omega = pi*(0:N)/N;
% p_1 = 5;
% 
% % Feedforward coefficient values.
% b0 = 1;
% bp = [0.1 0.5 0.9];
% 
% % Calculate the feedforward filter frequency response.
% Gf = b0 + transpose(bp)*exp(-1i*p_1*omega);
% 
% plot(omega/pi, 20*log10(abs(Gf)))
% grid
% xlabel('Normalized Radian Frequency');
% ylabel('Magnitude Response (dB)');
% legend('b_{p_1\alpha} = 0.1', 'b_{p_1\alpha} = 0.5', 'b_{p_1\alpha} = 0.9');
% 
%   


  % a)  part b
  %read keren
  [y,f_s]=audioread('KerenWithEcho.wav');
sound(y,f_s)
whos y

%Name           Size              Bytes  Class     Attributes

 % y         617400x1             4939200  double   

  %read roni
  [y1,f_s1]=audioread('RoniWithEcho.wav');
sound(y1,f_s1)
whos y1

%plot keren in sec
  t = (0:length(y) - 1)/f_s; %to get peak by sec we divide by the frequancy
figure(9);
      plot(t,y) 
      xlabel('Time (sec)')
      ylabel('Double-precision normalized samples ')
      title('keren with echo vs time [sec]');
      axis tight
 
 %roni
 figure(10);
      t1 = (0:length(y1) - 1)/f_s1; %to get peak by sec we divide by the frequancy
      plot(t1,y1) 
      xlabel('Time (sec)')
      ylabel('Double-precision normalized samples ')
      title('roni with echo vs time [sec]');
      axis tight
 
 
 
 % b)
 %autocorrelation can help verify the presence of delay


%Cross-correlation measures the similarity between a vector x and shifted (lagged) copies of a vector y as a function of the lag.
%If x and y have different lengths, the function appends zeros to the end of the shorter vector so it has the same length as the other.

%keren

[autocor,lags] = xcorr(y);% returns the lags at which the correlations are computed.
 
figure(11)
      plot(lags/f_s,autocor)
      xlabel('Lag/f_s (sec)')
      ylabel('autocorrelation')
      title('Autocorrelation keren')
      axis tight
      
      %p_keren=0.2*f_s

 
 %roni

[autocor1,lags1] = xcorr(y1);

figure(12)
      plot(lags1/f_s1,autocor1)
      xlabel('Lag/f_s (sec)')
      ylabel('autocorrelation')
      title('Autocorrelation roni')
      axis tight
      
      %p_roni=0.27*f_s;


 % c)  part b
 

 %keren
 %multiples echoes spaced R sampling periods 

 p=8820; %p=0.2*f_s
 den=[1,zeros(1,p-1),0.8];
  num=[1];
   w=0:0.01:pi;
    h=freqz(num,den,w);
    
    figure(13)
      subplot(2,1,1)
       plot(w/pi,abs(h),'b')
       grid on
       xlabel('\omega  [\pi*rad/sec]');
       ylabel('Magnitude Response ');
       title('frequency response of first filter (keren)');
     
    subplot(2,1,2)
       plot(w/pi,angle(h))
       ylabel('phase Response')
       xlabel('\omega  [\pi*rad/sec]');
  
  
  
 r=filter(num,den,y); % new keren song after filtering
  sound(r,f_s)
  
  
audiowrite('new_keren.wav',r,f_s);
  
  
  [y2,f_s2]=audioread('new_keren.wav');
sound(y2,f_s2);

figure(14);% plot keren 
subplot(2,1,1)
 
      t2 = (0:length(y2) - 1)/f_s2; %to get peak by sec we divide by the frequancy
      plot(t2,y2) 
	xlabel('Time (sec)')
      ylabel('Double-precision normalized samples ')
      title('keren without echo vs time [sec]');
      axis tight
 
 hold on
 subplot(2,1,2)
       plot(t,y) 
      xlabel('Time (sec)')
      ylabel('Double-precision normalized samples ')
      title('keren with echo vs time [sec]');
      axis tight
 
 
 
  %roni
 
 
 p_2=11907; %p_roni=0.27*f_s
 den1=[1,zeros(1,p_2-1),0.8];
  num1=[1];

   h_2=freqz(num1,den1,w);
   
   figure(15);% plot filter
 subplot(2,1,1)

    plot(w/pi,abs(h_2),'b')
    grid on
      xlabel('\omega  [\pi*rad/sec]');
      ylabel('Magnitude Response ');
    ylabel('magnitude response')
     title('frequency response of second filter (roni)')
     hold on
     
  subplot(2,1,2)
    plot(w/pi,angle(h_2))
   ylabel('phase Response')
   xlabel('\omega  [\pi*rad/sec]');
  






 r_=filter(num1,den1,y1) ; % new roni song after filtering
 
 
audiowrite('new_roni.wav',r_,f_s);
  
[y3,f_s3]=audioread('new_roni.wav');
sound(y3,f_s3);

  figure(16);%plot roni

subplot(2,1,1)
 
      t3 = (0:length(y3) - 1)/f_s3; %to get peak by sec we divide by the frequancy
      plot(t3,y3) 
      xlabel('Time (sec)')
      ylabel('Double-precision normalized samples ')
       title('roni without echo vs time [sec]');
      axis tight
 
 hold on
 subplot(2,1,2)
       plot(t1,y1) 
      xlabel('Time (sec)')
      ylabel('Double-precision normalized samples ')
      title('roni with echo vs time [sec]');
      axis tight
 

% adding two songs together
KerenAndRoniFiltered=cat(1,y2,y3);

soundsc(KerenAndRoniFiltered,f_s);

 

audiowrite('KerenAndRoniFiltered.wav',KerenAndRoniFiltered,f_s);


% done part b!!

%% PART C 
clc
clear all
%a

%breaking each signal to a matrix of 1 second each row
load('SignalsQ3.mat'); %load all signals
for k = 1:length(Signal1)/8000
    for i=1:8000
        x_t(k,i) = Signal1(i+(k-1)*8000);
    end
end

fs = 8000;
Ts = 1/fs;
example = x_t(1,:); %for this section we eill take an example of the first second 
[low_freq,high_freq] = domfreq(fs,example) %we get 2 values each for a dominent frequncy 

 t = 0:1/7999:1;
 figure(17)
 plot(t,example) 
 title('Single 1 , first second on time plane')
 xlabel('t [s]')
 ylabel('x(t)')
 
 %%b
 %we wrote a function calls commaornot that ind if a signal its a comma or not
 %we choos 2 example, 3rd second (word), and 4th second(comma)
 third = x_t(3,:);
 fourth = x_t(4,:);
answer = commaornot(third);
disp('the 3rd second is')
if answer == 0
    disp('a comma')
else
     disp('a word')
end
    
answer = commaornot(fourth);
disp('the 4th second is')
if answer == 0
    disp('a comma')
else
     disp('a word')
end

figure(18)
      plot(t,third) 
      title('Single 1 , third second on time plane')
      xlabel('t [s]')
      ylabel('x(t)')

  figure(19)
      plot(t,fourth) 
      title('Single 1 , fourth second on time plane')
      xlabel('t [s]')
      ylabel('x(t)')
 
 %%c
 
 for i = 1:length(x_t(:,1))
     answer = commaornot(x_t(i,:));
     if answer == 0
        decipher(i,1)=0;
        decipher(i,2)=0;
     else
        [low_freq,high_freq] = domfreq(fs,x_t(i,:));
        decipher(i,1)=low_freq;
        decipher(i,2)=high_freq;  
     end

 end
 
 %%e
 %e_a
 %first of all lets see what is the frequncy we need to filter so ill take
 %a random second and see it on frequncy domain

y=fft(Signal2(1:8000)); %calculating the fast foiurier transform 
p2 = abs(y/fs); %Compute the minus and plus spectrum 
p1 = p2(1:fs/2+1); %based on both sides we will compute one side of the tansform
p1(2:end-1) = 2*p1(2:end-1); %multypling by 2 all index not including the zero

f = 0:fs/2; 
figure(20)
      plot(f,p1) 
	title('random secon to see high frequncy')
      xlabel('f (Hz)')
      ylabel('amplitude')
      xlim([0 4500]);

%we need to filter 4kHz so we choose a low pass filter
%e_b
%i used filterDesigner to gnerate my_filter


%e_d

filter2 = filter(my_filter,Signal2);
%ploting the filterd and the non filterd signal 2 on time domain
t = 0:1/8000:(9-1/8000);
figure(21)
      plot(t,filter2) 
      title('filerd signal 2 time domain')
      xlabel('t [s]')
      ylabel('x(t)')

figure(22)
      plot(t,Signal2) 
      title('non filerd signal 2 time domain')
      xlabel('t [s]')
      ylabel('x(t)')

%ploting the zoom in on the  filterd and the non filterd signal 2 on time
%domain for better refrens
figure(23)
      plot(t,filter2) 
      title('zoom in on the filerd signal 2 time domain')
      xlabel('t [s]')
      ylabel('x(t)')
      xlim([1.75 2.25]);

figure(24)
      plot(t,Signal2) 
      title('zoom in on the non filerd signal 2 time domain')
      xlabel('t [s]')
      ylabel('x(t)')
      xlim([1.75 2.25]);

% ploting the filterd and the non filterd signal 2 on the frequncy domain

y=fft(filter2); %calculating the fast foiurier transform of the  filterd signal 2
p2 = abs(y/fs); %Compute the minus and plus spectrum 
p1 = p2(1:fs/2+1); %based on both sides we will compute one side of the tansform
p1(2:end-1) = 2*p1(2:end-1); %multypling by 2 all index not including the zero

f = 0:fs/2; 
figure(25)   % plotin the signal by its frequncy
      plot(f,p1) 
      title('filterd signal 2 on the frequncy domain')
      xlabel('f (Hz)')
      ylabel('amplitude')
      xlim([0 4500]);

y=fft(Signal2); %calculating the fast foiurier transform of the  filterd signal 2
p2 = abs(y/fs); %Compute the minus and plus spectrum 
p1 = p2(1:fs/2+1); %based on both sides we will compute one side of the tansform
p1(2:end-1) = 2*p1(2:end-1); %multypling by 2 all index not including the zero

f = 0:fs/2; 
figure(26)   % plotin the signal by its frequncy
      plot(f,p1) 
      title('non filterd signal 2 on the frequncy domain')
      xlabel('f (Hz)')
      ylabel('amplitude')
      xlim([0 4500]);
 % ploting the filterd and the non filterd signal 2 on the frequncy domain
% only on 1 second from the 9, for better refrens

y=fft(filter2(24001:32000)); %calculating the fast foiurier transform of the fourth second filterd signal 2 
p2 = abs(y/fs); %Compute the minus and plus spectrum 
p1 = p2(1:fs/2+1); %based on both sides we will compute one side of the tansform
p1(2:end-1) = 2*p1(2:end-1); %multypling by 2 all index not including the zero

f = 0:fs/2; 
figure(27)   % plotin the signal by its frequncy
      plot(f,p1) 
      title('filterd signal 2 on the frequncy domain')
      xlabel('f (Hz)')
      ylabel('amplitude')
      xlim([0 4500]);

y=fft(Signal2(24001:32000)); %calculating the fast foiurier transform of the fourth second non filterd signal 2
p2 = abs(y/fs); %Compute the minus and plus spectrum 
p1 = p2(1:fs/2+1); %based on both sides we will compute one side of the tansform
p1(2:end-1) = 2*p1(2:end-1); %multypling by 2 all index not including the zero

f = 0:fs/2; 
figure (28)  % plotin the signal by its frequncy
      plot(f,p1) 
      title('non filterd signal 2 on the frequncy domain')
      xlabel('f (Hz)')
      ylabel('amplitude')
      xlim([0 4500]);

%using the same loop as last part to see the code
for k = 1:length(filter2)/8000
    for i=1:8000
        x_t2(k,i) = filter2(i+(k-1)*8000);
    end
end

  for i = 1:length(x_t2(:,1))
     answer = commaornot(x_t2(i,:));
     if answer == 0
        decipher2(i,1)=0;
        decipher2(i,2)=0;
     else
        [low_freq,high_freq] = domfreq(fs,x_t2(i,:));
        decipher2(i,1)=low_freq;
        decipher2(i,2)=high_freq;  
     end

 end

 %%f
 %f_a
 % first lets see what is the angle to filter 800 Hz noise for fs=8000 Hz 
 % sampling. fs/fn=2pi/theta =>theta = (1/5)pi 
 z1 = exp(j*(1/5)*pi); 
 z2 = exp(-j*(1/5)*pi); 
 Bz = poly([z1 z2]); 
 Z = roots(Bz); 

 
 Az = poly([0.9*z1 0.9*z2]);     
 P = roots(Az);     
 figure(29);     
      zplane(Z,P);     
      title('zeroes and poles map ' );     
 figure(30);     
      freqz(Bz,Az);     
      title('Amplitude and Phase graph' );     
 

 t = 0:(1/8000):length(Signal3)/8000;
 t = t(2:1:end);
 filter3 = filter(Bz,Az,Signal3); 
 figure (31)
      plot(t,Signal3); 
      title('Signal 3 800Hz non filtered') 
      xlabel('t [sec]'); 
      ylabel('Amplitude');  
      grid on  
 
 figure(32) 
      plot(t,filter3); 
      title('Signal 3 800Hz filtered') 
      xlabel('t [sec]'); 
      ylabel('Amplitude');  
      grid on  
 
 
y=fft(filter3(24001:32000)); %calculating the fast foiurier transform of the fourth second filterd signal 2 
p2 = abs(y/fs); %Compute the minus and plus spectrum 
p1 = p2(1:fs/2+1); %based on both sides we will compute one side of the tansform
p1(2:end-1) = 2*p1(2:end-1); %multypling by 2 all index not including the zero

f = 0:fs/2; 
figure(33)   % plotin the signal by its frequncy
      plot(f,p1) 
      title('filterd signal 3 on the frequncy domain')
      xlabel('f (Hz)')
      ylabel('amplitude')
      xlim([0 4500]);

y=fft(Signal3(24001:32000)); %calculating the fast foiurier transform of the fourth second non filterd signal 2
p2 = abs(y/fs); %Compute the minus and plus spectrum 
p1 = p2(1:fs/2+1); %based on both sides we will compute one side of the tansform
p1(2:end-1) = 2*p1(2:end-1); %multypling by 2 all index not including the zero

f = 0:fs/2; 
figure(34)   % plotin the signal by its frequncy
      plot(f,p1) 
      title('non filterd signal 3 on the frequncy domain')
      xlabel('f (Hz)')
      ylabel('amplitude')
      xlim([0 4500]);

for k = 1:length(filter3)/8000
    for i=1:8000
        x_t3(k,i) = filter3(i+(k-1)*8000);
    end
end

  for i = 1:length(x_t3(:,1))
     answer = commaornot(x_t3(i,:));
     if answer == 0
        decipher3(i,1)=0;
        decipher3(i,2)=0;
     else
        [low_freq,high_freq] = domfreq(fs,x_t3(i,:));
        decipher3(i,1)=low_freq;
        decipher3(i,2)=high_freq;  
     end

 end
 %g
 %first lets see the 3 second of the first signal
 
 
y=fft(Signal1(16001:24000)); %calculating the fast foiurier transform of the 3rd second of signal 1
p2 = abs(y/fs); %Compute the minus and plus spectrum 
p1 = p2(1:fs/2+1); %based on both sides we will compute one side of the tansform
p1(2:end-1) = 2*p1(2:end-1); %multypling by 2 all index not including the zero

f = 0:fs/2; 
figure(35)  % plotin the signal by its frequncy
      plot(f,p1) 
      title('3rd second of signal 1')
      xlabel('f (Hz)')
      ylabel('amplitude')
      xlim([0 4500]);

fs4 = 2000;
y=fft(Signal4(4001:6000)); %calculating the fast foiurier transform of the 3rd second of signal 4
p2 = abs(y/fs4); %Compute the minus and plus spectrum 
p1 = p2(1:fs4/2+1); %based on both sides we will compute one side of the tansform
p1(2:end-1) = 2*p1(2:end-1); %multypling by 2 all index not including the zero

f = 0:fs4/2; 
figure(36)   % plotin the signal by its frequncy
      plot(f,p1) 
      title('3rd second of signal 4')
      xlabel('f (Hz)')
      ylabel('amplitude')
      xlim([0 4500]);

%ploting on time domain secon 2-3 (the 3rd second)
t = 0:1/8000:(11-1/8000);
figure(37)
      plot(t,Signal1) 
      title('signal 1 time domain')
      xlabel('t [s]')
      ylabel('x(t)')
      xlim([2 3]);

t = 0:1/2000:(11-1/2000);
figure(38)
      plot(t,Signal4) 
      title('signal 4 time domain')
      xlabel('t [s]')
      ylabel('x(t)')
      xlim([2 3]);


% zoom in time domain secon 2-3 (the 3rd second) to 2.5s-2.7s

t = 0:1/8000:(11-1/8000);
figure(39)
      plot(t,Signal1) 
      title('zoom in signal 1 time domain')
      xlabel('t [s]')
      ylabel('x(t)')
      xlim([2.5 2.7]);

t = 0:1/2000:(11-1/2000);
figure(40)
      plot(t,Signal4) 
      title('zoom in signal 4 time domain')
      xlabel('t [s]')
      ylabel('x(t)')
      xlim([2.5 2.7]);