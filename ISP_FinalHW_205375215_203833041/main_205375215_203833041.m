
clear all

%% Q_1

T0 = 10;
w0 = 2*pi/T0;
t = linspace (-5, 5,5000); %f_s=5000/10=500
sq=[zeros(1,500),4*ones(1,500),ones(1,500),4*ones(1,500),zeros(1,1000)... %the real signal for refrense
    ,-1*ones(1,500),-4*ones(1,500),-1*ones(1,500),zeros(1,500)];

Xf_1=X_t_FFT(1,t);

figure;
plot(t,sq,'r','LineWidth',2)
hold on;

plot(t,real(Xf_1),'b','Linewidth',0.5);
title(' \fontsize{16} Reconstructed signal by 1 coefficient')
xlabel('\fontsize{14} Time [s]')
ylabel(' \fontsize{14} x(t) and FS')
legend('\fontsize{12} x(t)',' \fontsize{12} fourier series of x(t) n=1','location','Northeast');
grid on


Xf_10=X_t_FFT(10,t);
figure;
plot(t,sq,'r','LineWidth',2)
hold on;
plot(t,real(Xf_10),'b','Linewidth',0.5);
title('\fontsize{16} Reconstructed signal by 10 coefficients')
xlabel('\fontsize{14} Time [s]')
ylabel('\fontsize{14} x(t) and FS')
legend('\fontsize{12} x(t)',' \fontsize{12} fourier series of x(t) n=10','location','Northeast');
grid on


Xf_100=X_t_FFT(100,t);
figure;
plot(t,sq,'r','LineWidth',2)
hold on;
plot(t,real(Xf_100),'b','Linewidth',0.5);
title('\fontsize{16} Reconstructed signal by 100 coefficients')
xlabel(' \fontsize{14}  Time [s]')
ylabel('\fontsize{14} x(t) and FS')
legend(' \fontsize{12}  x(t) ',' \fontsize{12} fourier series of x(t) n=100','lo00cation','Northeast');
grid on

%error calculation

ER_n_1= real(Xf_1); % So as to get rid of -0.000000000i (imaginary) factor
    Max_1 = max(ER_n_1);
    Min_1 = min(ER_n_1); 
    M_1 = max(abs(Max_1),abs(Min_1)); % Maximum error
    L_1=4-M_1; % length of the relative overshot
    Overshoot_n_1 = ((L_1)/4)*100; % Overshoot calculation

ER_n_2 = real(Xf_10);
    Max_10 = max(ER_n_2);
    Min_10 = min(ER_n_2); 
    M_10 = max(abs(Max_10),abs(Min_10)); 
    L_10=M_10-4; % length of the relative overshot
    Overshoot_n_10 = ( L_10/4)*100; 

    
ER_n_100= real(Xf_100); 
    Max_100 = max(ER_n_100);
    Min_100 = min(ER_n_100); 
    M_100 = max(abs(Max_100),abs(Min_100)); % Maximum error
    L_100=M_100-4; % length of the relative overshot
    Overshoot_n_100 = (L_100/4)*100; % Overshoot calculation

%% Q_2

syms f

H_f=(-1i)*(sign(2*pi*f-pi)+sign(2*pi*f+pi));

H_f_mgn = abs(H_f);
H_f_arg= angle(H_f);

figure
fplot(f,H_f_mgn,'b','Linewidth',4)
title('\fontsize{16} \mid H(f) \mid')
xlabel(' \fontsize{14}  frequancy [Hz]')
ylabel('\fontsize{14} \mid H(f) \mid')
legend(' \fontsize{12} \mid H(f) \mid ')
ylim([-1 3]);
grid on

figure
fplot(f,H_f_arg,'r','Linewidth',4)
title('\fontsize{16} \langle H(f) ')
xlabel(' \fontsize{14}  frequancy [Hz]')
ylabel('\fontsize{14} \mid H(f) \mid')
legend(' \fontsize{12} \langle H(f)','lo00cation','Northeast');
ylim([-3 3]);
grid on


%% Q_3

% clear all
% signal 1 ----------------------------------------------------------------
[y,fs] = audioread('Signal1.wav');
n = length(y);              % number of samples
Y = fft(y,n);               % Fourier transform of y
F = ((0:1/n:1-1/n)*fs);     % Frequency vector
amp = abs(Y); 

%ploting of signal 1 and its zoom in
figure ;
plot(F, amp);
grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude [dB]');
title('Magnitude spectrum of signal 1');

figure ;
plot(F, amp);
grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude [dB]');
title('Amplitude spectrum of signal 1 zoom in');
xlim([0 10000])
ylim([0 30000])

% signal 2 ----------------------------------------------------------------
[y2,fs2] = audioread('Signal2.wav');  
n2 = length(y2);               % number of samples
Y2 = fft(y2,n2);               % Fourier transform of y2
F2 = ((0:1/n2:1-1/n2)*fs2);    % Frequency vector
amp2 = abs(Y2); 

%ploting of signal 2 and its zoom in
figure ;
plot(F2, amp2);
grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude [dB]');
title('Amplitude spectrum of signal 2');

figure ;
plot(F2, amp2);
grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude [dB]');
title('Amplitude spectrum of signal 2 zoom in');
xlim([0 10000])
ylim([0 30000])

figure ;
plot(F2, amp2,'r');
grid on;
hold on;
plot(F, amp,'b');
xlabel('Frequency [Hz]');
ylabel('Amplitude [dB]');
title('Amplitude spectrum of signals 1+2 zoom in');
legend('signal 2','signal 1','lo00cation','Northeast');
xlim([0 5000])
ylim([0 30000])


%% Q_4 
% alef 
% clear all
  % y coordinate

trail1=load('Trial1NoPerturbation.mat');

theta=linspace(0,2*pi,100);
r=2.5;

figure
hold on 
plot(r*cos(theta),r*sin(theta),'k','linewidth',2);
xlim([-6 6]);
ylim([-6 6]);
hold on
plot(trail1.Psm1X,trail1.Psm1Y,'r','linewidth',1.5)
title('\fontsize{16} scissors path Trial 1')
xlabel(' \fontsize{16}  x [cm]')
ylabel('\fontsize{16} y  [cm]')
 legend('\fontsize{12} Desirable Circle',' \fontsize{12}  gauza ',' \fontsize{12} scissors path','lo00cation','Northeast');
 hold on 
 rectangle('Position',[-5 -5 10 10])
 grid on

 
 trail2=load('Trial2PerturbationA.mat');
 
figure
 plot(r*cos(theta),r*sin(theta),'k','linewidth',2);
xlim([-6 6]);
ylim([-6 6]);

hold on
plot(trail2.Psm1X,trail2.Psm1Y,'r','linewidth',1.5);
title('\fontsize{16} scissors path Trial 2')
xlabel(' \fontsize{16}  x [cm]')
ylabel('\fontsize{16} y  [cm]')
 legend('  \fontsize{12}  gauza ',' \fontsize{12} scissors path','lo00cation','Northeast');
hold on 
 rectangle('Position',[-5 -5 10 10]);
 grid on
 
 
 trail3=load('Trial3PerturbationB.mat');
 
figure
 plot(r*cos(theta),r*sin(theta),'k','linewidth',2);
xlim([-6 6]);
ylim([-6 6]);

hold on
plot(trail3.Psm1X,trail3.Psm1Y,'r','linewidth',1.5);
title('\fontsize{16} scissors path Trial 3')
xlabel(' \fontsize{16}  x [cm]')
ylabel('\fontsize{16} y  [cm]')
 legend(' \fontsize{12}  gauza ',' \fontsize{12} scissors path','lo00cation','Northeast');
hold on 
 rectangle('Position',[-5 -5 10 10])
 grid on
 
 
% bet 
% a
%rad 1 ------------------------------------
 rad_1 = sqrt((trail1.Psm1X).^2+(trail1.Psm1Y).^2); %radius of circle 
 
 L_1 = length(trail1.Psm1X);
 FFT_t_1=fft(rad_1,L_1);

 T0 = 1/100;
f_s=1/T0;

time_1= T0*(L_1-1); %convert to sec

t_1= 0:0.01:time_1;
t_1= t_1'; %transpose our time vector

F_vec_1 = ((0:1/L_1:1-1/L_1)*f_s);     % Frequency vector
amp_1 = abs(FFT_t_1); 

figure 
subplot(2,1,1); %time domain

      plot(t_1,rad_1)
      title('\fontsize{16} Time domain Trial 1')
      xlabel(' \fontsize{16} Time [sec]')
      ylabel('\fontsize{16}  Psm1Radius [cm]')
      %legend('\fontsize{12} Trial 1 Psm1Radius in Time domain')
      grid on
     
    
      hold on
subplot(2,1,2) %frequancy domain
  plot(F_vec_1, amp_1);
      grid on;
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} frequncy Amplitude Trial 1');
      %legend(' \fontsize{12} Trial 1 Psm1Radius in Frequency domain');
      xlim([-10 110]);

%rad 2 ------------------------------------
rad_2 = sqrt((trail2.Psm1X).^2+(trail2.Psm1Y).^2); 

L_2 = length(trail2.Psm1X);
time_2 = T0*(L_2-1);
t_2= 0:0.01:time_2;
t_2= t_2';
 FFT_t_2=fft(rad_2,L_2);

F_vec_2 = ((0:1/L_2:1-1/L_2)*f_s);     % Frequency vector
amp_2 = abs(FFT_t_2); 

figure
subplot(2,1,1); %time domain

       plot(t_2,rad_2)
      title('\fontsize{16} Time domain Trial 2')
      xlabel(' \fontsize{16} Time [sec]')
      ylabel('\fontsize{16}  Psm1Radius [cm]')
      %legend('\fontsize{12} Trial 2 Psm1Radius in Time domain')
      grid on
    
      hold on
subplot(2,1,2) %frequancy domain
  plot(F_vec_2, amp_2);
      grid on;
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} Frequncy Amplitude Trial 2');
      %legend(' \fontsize{12} Trial 2 Psm1Radius in Frequency domain');
      xlim([-10 110]);
 
      figure 
      plot(F_vec_2, amp_2);
      grid on;
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} Frequncy Amplitude Trial 2 zoom in ');
      xlim([-10 10]);



%rad 3 ------------------------------------ 
rad_3 = sqrt((trail3.Psm1X).^2+(trail3.Psm1Y).^2); %calculate the lengrh of the radius vector

L_3 = length(trail3.Psm1X);
time_3 = T0*(L_3-1);
t_3= 0:0.01:time_3;
t_3= t_3';
  
 FFT_t_3=fft(rad_3,L_3);
 
time_3= T0*(L_3-1); %convert to sec

F_vec_3 = ((0:1/L_3:1-1/L_3)*f_s);     % Frequency vector
amp_3 = abs(FFT_t_3); 

figure
subplot(2,1,1); %time domain

      plot(t_3,rad_3)
      title('\fontsize{16} Time domain Trial 3')
      xlabel(' \fontsize{16} Time [sec]')
      ylabel('\fontsize{16}  Psm1Radius [cm]')
     % legend('\fontsize{12} Trial 3 Psm1Radius in frequancy domain')
      grid on
    
      hold on
      subplot(2,1,2) %frequancy domain
      plot(F_vec_3, amp_3);
      grid on;
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} Frequncy Amplitude Trial 3');
      %legend(' \fontsize{12} Trial 3 Psm1Radius in time domain');
      xlim([-10 110]);
      
      figure
      plot(F_vec_3, amp_3);
      grid on;
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} Frequncy Amplitude Trial 3 zoom in');
      %legend(' \fontsize{12} Trial 3 Psm1Radius in time domain');
      xlim([-10 10]);
      
      %baseline 
      % calculate the baseline before advance
      % we want to remove 1 Hz disturbance 
      % for that purpose we will define
      
      f_remove=1; %Hz  
         %our beloved notch
    omega=2*pi*f_remove*T0;
    p=exp(1i*omega);
    b=poly([p conj(p)]); 
    a=poly([0.995*p 0.995*conj(p)]);
figure 
freqz(b,a);
title('Amplitude and Phase graph' );
xlim([0 0.2])
%filtering

y_1=filter(b,a,rad_1);            %The Baseline

FFT_y_1=fft(y_1,L_1);               %The  fft for the Baseline
F_vec_1 = ((0:1/L_1:1-1/L_1)*f_s);  % Frequency vector
amp_y_1 = abs(FFT_y_1);             


figure ; %time domain
subplot(2,1,1)
      plot(t_1,rad_1,'LineWidth',1);
      hold on
      plot(t_1,y_1);
      title('\fontsize{16} Time domain Trial 1')
      xlabel(' \fontsize{16} Time [sec]')
      ylabel('\fontsize{16}  Psm1Radius [cm]')
      legend('\fontsize{12} Psm Radius','\fontsize{12} Baseline')
      grid on
      hold on
 
subplot(2,1,2) %Isolated Perturbation domain
det_1 = rad_1-y_1; % isolate the baseline as x axsis
  plot(t_1,det_1);
      grid on;
      xlabel('\fontsize{16} time [sec]');     
      ylabel('\fontsize{16} Isolated Perturbation [cm]');
      title('\fontsize{16} Isolated Perturbation Trial 1');
      legend(' \fontsize{12} Isolated Perturbation');
      
 
      %plot freq domain to check the filtration
 figure;
  subplot(2,1,1)
 plot(F_vec_1, amp_y_1);
      grid on;
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} Trial 1 Baseline in frequancy domain');
      xlim([-10 110]);
      hold on
      
FFT_iso_1=fft(det_1,L_1);
F_vec_1 = ((0:1/L_1:1-1/L_1)*f_s);     % Frequency vector
amp_iso_1 = abs(FFT_iso_1);

      subplot(2,1,2);
      plot(F_vec_1, amp_iso_1);
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} Trial 1 Isolated Perturbation in frequancy domain');
      xlim([-10 110]);
      grid on
      
figure
      plot(F_vec_1, amp_iso_1);
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{14} Trial 1 Isolated Perturbation in frequancy domain zoom in');
      xlim([-1 10]);
      grid on



y_2=filter(b,a,rad_2);

FFT_y_2=fft(y_2,L_2);
F_vec_2 = ((0:1/L_2:1-1/L_2)*f_s);     % Frequency vector
amp_y_2 = abs(FFT_y_2); 


figure ; %time domain
subplot(2,1,1)
      plot(t_2,rad_2);
      hold on
      plot(t_2,y_2);
      title('\fontsize{16} Time domain Trial 2')
      xlabel(' \fontsize{16} Time [sec]')
      ylabel('\fontsize{16}  Psm2Radius [cm]')
      legend('\fontsize{12} Psm Radius','\fontsize{12} Baseline ')
      grid on;
     
    
      hold on
 
subplot(2,1,2) %Isolated Perturbation domain
det_2 = rad_2-y_2; % isolate the baseline as x axsis
  plot(t_2,det_2);
      grid on;
      xlabel('\fontsize{16} time [sec]');     
      ylabel('\fontsize{16} Isolated Perturbation [cm]');
      title('\fontsize{16} Isolated Perturbation Trial 2');
    
 
      %plot freq domain to check the filtration
 figure;
  subplot(2,1,1)
 plot(F_vec_2, amp_y_2);
      grid on;
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} Trial 2 Baseline in frequancy domain');
      xlim([-10 110]);
      hold on
      
FFT_iso_2=fft(det_2,L_2);
F_vec_2 = ((0:1/L_2:1-1/L_2)*f_s);     % Frequency vector
amp_iso_2 = abs(FFT_iso_2);

      subplot(2,1,2);
      plot(F_vec_2, amp_iso_2);
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} Trial 2 Isolated Perturbation in frequancy domain');
      xlim([-10 110]);
      grid on;
      
      
      
    figure % close up on Isolated Perturbation
      plot(F_vec_2, amp_iso_2);
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{14} Trial 2 Isolated Perturbation in frequancy domain zoom in ');
      xlim([-1 10]);
      grid on;


y_3=filter(b,a,rad_3);
FFT_y_3=fft(y_3,L_3);
F_vec_3 = ((0:1/L_3:1-1/L_3)*f_s);     % Frequency vector
amp_y_3 = abs(FFT_y_3); 


figure ; %time domain
subplot(2,1,1)
      plot(t_3,rad_3);
      hold on
      plot(t_3,y_3);
      title('\fontsize{16} Time domain Trial 3')
      xlabel(' \fontsize{16} Time [sec]')
      ylabel('\fontsize{16}  Psm3Radius [cm]')
      legend('\fontsize{12} Psm Radius','\fontsize{12} Baseline ')
      grid on;
     
    
      hold on
 
subplot(2,1,2) %Isolated Perturbation domain
det_3 = rad_3-y_3; % isolate the baseline as x axsis
  plot(t_3,det_3);
      grid on;
      xlabel('\fontsize{16} time [sec]');     
      ylabel('\fontsize{16} Isolated Perturbation [cm]');
      title('\fontsize{16} Isolated Perturbation Trial 3');
    
 
      %plot freq domain to check the filtration
 figure;
  subplot(2,1,1)
 plot(F_vec_3, amp_y_3);
      grid on;
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} Trial 3 Baseline in frequancy domain');
      xlim([-10 110]);
      hold on
      
FFT_iso_3=fft(det_3,L_3);
F_vec_3 = ((0:1/L_3:1-1/L_3)*f_s);     % Frequency vector
amp_iso_3 = abs(FFT_iso_3);

      subplot(2,1,2);
      plot(F_vec_3, amp_iso_3);
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{16} Trial 3 Isolated Perturbation in frequancy domain');
      xlim([-10 110]);
     grid on
      
      %close up
      figure
      plot(F_vec_3, amp_iso_3);
      xlabel('\fontsize{16} Frequency [Hz]');     
      ylabel('\fontsize{16} Amplitude [dB]');
      title('\fontsize{14} Trial 3 Isolated Perturbation in frequancy domain zoom im');
      xlim([-1 10]);
      grid on
      
      
      
      %gimel
      %a
      
      %trial_1
      
      
      
      %lets start with compare the index of thr relevent vector
      
 for i_1=1:length(trail1.RightHalfInd)
     right_1(i_1,1)=trail1.Psm1X(trail1.RightHalfInd(i_1));
     right_1(i_1,2)=trail1.Psm1Y(trail1.RightHalfInd(i_1));
 end
 
 
  for i_1=1:length(trail1.LeftHalfInd)
     left_1(i_1,1)=trail1.Psm1X(trail1.LeftHalfInd(i_1));
     left_1(i_1,2)=trail1.Psm1Y(trail1.LeftHalfInd(i_1));
 end



%the baseline is the radius of the circle, all we need is its angle so well
%compare each index to its corolated angle as given
argu_1 = trail1.Psm1Y./trail1.Psm1X;
theta_1 = atan(argu_1);

%autamaticly x define to be possitive in this algorithem, so we make sure
%that the sign of each x componnent will be as the original
for i_1=1:length(trail1.Psm1X)
    if trail1.Psm1X(i_1) >= 0
       x_1(i_1) = cos(theta_1(i_1))*y_1(i_1);
    else
       x_1(i_1) = -cos(theta_1(i_1))*y_1(i_1);
    end
end

for i_1=1:length(trail1.Psm1Y)
    if trail1.Psm1X(i_1) >= 0
       y_1(i_1) = sin(theta_1(i_1))*y_1(i_1);
    else
       y_1(i_1) = -sin(theta_1(i_1))*y_1(i_1);
    end
end
       
 % 
 for i_1=1:length(trail1.RightHalfInd)
     rightx_1(i_1)=x_1(trail1.RightHalfInd(i_1));
     righty_1(i_1)=y_1(trail1.RightHalfInd(i_1));
 end
 

  for i_1=1:length(trail1.LeftHalfInd)
     leftx_1(i_1)=x_1(trail1.LeftHalfInd(i_1));
     lefty_1(i_1)=y_1(trail1.LeftHalfInd(i_1));
  end

  

figure
plot(r*cos(theta),r*sin(theta),'k','linewidth',2);
hold on
plot(leftx_1,lefty_1,'r','linewidth',1.5)
xlim([-6 6])
ylim([-6 6])
hold on
plot(rightx_1,righty_1,'r','linewidth',1.5)
hold on
plot(right_1(:,1),right_1(:,2),'b','linewidth',0.75)
hold on
plot(left_1(:,1),left_1(:,2),'b','linewidth',0.75)
hold on
rectangle('Position',[-5 -5 10 10]);
xlabel('\fontsize{16} x [cm]');     
      ylabel('\fontsize{16} y  [cm]');
      title('\fontsize{16} scissors''s path and  Filtered Scissors''s Path Trail 1  ');
legend('\fontsize{12} Desirable Circle',' \fontsize{12}Filtered  scissors''s path',' \fontsize{12}Filtered  scissors''s path','\fontsize{12} scissors''s path');
grid on


      %trial_2
      %lets start with compare the index of thr relevent vector
      
 for i_2=1:length(trail2.RightHalfInd)
     right_2(i_2,1)=trail2.Psm1X(trail2.RightHalfInd(i_2));
     right_2(i_2,2)=trail2.Psm1Y(trail2.RightHalfInd(i_2));
 end
 
 
  for i_2=1:length(trail2.LeftHalfInd)
     left_2(i_2,1)=trail2.Psm1X(trail2.LeftHalfInd(i_2));
     left_2(i_2,2)=trail2.Psm1Y(trail2.LeftHalfInd(i_2));
  end


%the baseline is the radius of the circle, all we need is its angle so well
%compare each index to its corolated angle as given
argu_2 = trail2.Psm1Y./trail2.Psm1X;
theta_2 = atan(argu_2);

%autamaticly x define to be possitive in this algorithem, so we make sure
%that the sign of each x componnent will be as the original
for i_2=1:length(trail2.Psm1X)
    if trail2.Psm1X(i_2) >= 0
       x_2(i_2) = cos(theta_2(i_2))*y_2(i_2);
    else
       x_2(i_2) = -cos(theta_2(i_2))*y_2(i_2);
    end
end

for i_2=1:length(trail2.Psm1Y)
    if trail2.Psm1X(i_2) >= 0
       y_2(i_2) = sin(theta_2(i_2))*y_2(i_2);
    else
       y_2(i_2) = -sin(theta_2(i_2))*y_2(i_2);
    end
end
       


 % 
 for i_2=1:length(trail2.RightHalfInd)
     rightx_2(i_2)=x_2(trail2.RightHalfInd(i_2));
     righty_2(i_2)=y_2(trail2.RightHalfInd(i_2));
 end
 

  for i_2=1:length(trail2.LeftHalfInd)
     leftx_2(i_2)=x_2(trail2.LeftHalfInd(i_2));
     lefty_2(i_2)=y_2(trail2.LeftHalfInd(i_2));
  end

figure
plot(r*cos(theta),r*sin(theta),'k','linewidth',2);
hold on
plot(leftx_2,lefty_2,'r','linewidth',1.5)
xlim([-6 6])
ylim([-6 6])
hold on
plot(rightx_2,righty_2,'r','linewidth',1.5)
hold on
plot(right_2(:,1),right_2(:,2),'b','linewidth',1.5)
hold on
plot(left_2(:,1),left_2(:,2),'b','linewidth',1.5)
hold on
rectangle('Position',[-5 -5 10 10]);
xlabel('\fontsize{16} x [cm]');     
      ylabel('\fontsize{16} y  [cm]');
      title('\fontsize{16} scissors''s path and  Filtered Scissors''s Path Trail 2  ');
legend('\fontsize{12} Desirable Circle','\fontsize{12} Filtered Scissors''s Path',' \fontsize{12}Filtered  scissors''s path','\fontsize{12} scissors''s path');
grid on



     %trial_3
      %lets start with compare the index of thr relevent vector
      
 for i_3=1:length(trail3.RightHalfInd)
     right_3(i_3,1)=trail3.Psm1X(trail3.RightHalfInd(i_3));
     right_3(i_3,2)=trail3.Psm1Y(trail3.RightHalfInd(i_3));
 end
 
 
  for i_3=1:length(trail3.LeftHalfInd)
     left_3(i_3,1)=trail3.Psm1X(trail3.LeftHalfInd(i_3));
     left_3(i_3,2)=trail3.Psm1Y(trail3.LeftHalfInd(i_3));
 end



%the baseline is the radius of the circle, all we need is its angle so well
%compare each index to its corolated angle as given
argu_3 = trail3.Psm1Y./trail3.Psm1X;
theta_3 = atan(argu_3);

%autamaticly x define to be possitive in this algorithem, so we make sure
%that the sign of each x componnent will be as the original
for i_3=1:length(trail3.Psm1X)
    if trail3.Psm1X(i_3) >= 0
       x_3(i_3) = cos(theta_3(i_3))*y_3(i_3);
    else
       x_3(i_3) = -cos(theta_3(i_3))*y_3(i_3);
    end
end

for i_3=1:length(trail3.Psm1Y)
    if trail3.Psm1X(i_3) >= 0
       y_3(i_3) = sin(theta_3(i_3))*y_3(i_3);
    else
       y_3(i_3) = -sin(theta_3(i_3))*y_3(i_3);
    end
end
       


 % 
 for i_3=1:length(trail3.RightHalfInd)
     rightx_3(i_3)=x_3(trail3.RightHalfInd(i_3));
     righty_3(i_3)=y_3(trail3.RightHalfInd(i_3));
 end
 

  for i_3=1:length(trail3.LeftHalfInd)
     leftx_3(i_3)=x_3(trail3.LeftHalfInd(i_3));
     lefty_3(i_3)=y_3(trail3.LeftHalfInd(i_3));
  end

figure

plot(r*cos(theta),r*sin(theta),'k','linewidth',2);
hold on
plot(leftx_3,lefty_3,'r','linewidth',1.5)
xlim([-6 6])
ylim([-6 6])
hold on
plot(rightx_3,righty_3,'r','linewidth',1.5)
hold on
plot(right_3(:,1),right_3(:,2),'b','linewidth',1.5)
hold on
plot(left_3(:,1),left_3(:,2),'b','linewidth',1.5)
hold on
rectangle('Position',[-5 -5 10 10]);
xlabel('\fontsize{16} x [cm]');     
      ylabel('\fontsize{16} y  [cm]');
      title('\fontsize{16} scissors''s path and  Filtered Scissors''s Path Trail 3  ');
legend('\fontsize{12} Desirable Circle','\fontsize{12} Filtered Scissors''s Path',' \fontsize{12}Filtered  scissors''s path','\fontsize{12} scissors''s path');
grid on

% b
% average baseline dustance trail 2
baseright_1 = sqrt(rightx_1.^2 + righty_1.^2);  %only the right side of the baseline
baseleft_1 = sqrt(leftx_1.^2 + lefty_1.^2); %only the left side of the baseline
base_1 = horzcat(1,baseright_1,baseleft_1);%combining the two vectors into 1
base_1dis = abs(base_1(2:end)-2.5);  % the distance to the reak gauza circle while ignoring the first valur which add'd after horzcat 
baseav_dis_1 = mean(base_1dis); % baseline average distance
disp(baseav_dis_1)


realright_1 = sqrt(right_1(:,1).^2 + right_1(:,2).^2 )';  %only the right side of the real movment
realleft_1 = sqrt(left_1(:,1).^2 + left_1(:,2).^2)'; %only the left side of the real movment
real_1 = horzcat(1,realright_1,realleft_1);%combining the two vectors into 1
real_1dis = abs(real_1(2:end)-2.5);  % the distance to the reak gauza circle while ignoring the first valur which add'd after horzcat 
realav_dis_1 = mean(real_1dis); % baseline average distance
disp(realav_dis_1)



% average baseline dustance trail 2
baseright_2 = sqrt(rightx_2.^2 + righty_2.^2);  %only the right side of the baseline
baseleft_2 = sqrt(leftx_2.^2 + lefty_2.^2); %only the left side of the baseline
base_2 = horzcat(1,baseright_2,baseleft_2);%combining the two vectors into 1
base_2dis = abs(base_2(2:end)-2.5);  % the distance to the reak gauza circle while ignoring the first valur which add'd after horzcat 
baseav_dis_2 = mean(base_2dis); % baseline average distance
disp(baseav_dis_2)


realright_2 = sqrt(right_2(:,1).^2 + right_2(:,2).^2 )';  %only the right side of the real movment
realleft_2 = sqrt(left_2(:,1).^2 + left_2(:,2).^2)'; %only the left side of the real movment
real_2 = horzcat(1,realright_2,realleft_2);%combining the two vectors into 1
real_2dis = abs(real_2(2:end)-2.5);  % the distance to the reak gauza circle while ignoring the first valur which add'd after horzcat 
realav_dis_2 = mean(real_2dis); % baseline average distance
disp(realav_dis_2)

% average baseline dustance trail 3
baseright_3 = sqrt(rightx_3.^2 + righty_3.^2);  %only the right side of the baseline
baseleft_3 = sqrt(leftx_3.^2 + lefty_3.^2); %only the left side of the baseline
base_3 = horzcat(1,baseright_3,baseleft_3);%combining the two vectors into 1
base_3dis = abs(base_3(2:end)-2.5);  % the distance to the reak gauza circle while ignoring the first valur which add'd after horzcat 
baseav_dis_3 = mean(base_3dis) ;% baseline average distance
disp(baseav_dis_3)

realright_3 = sqrt(right_3(:,1).^2 + right_3(:,2).^2 )';  %only the right side of the real movment
realleft_3 = sqrt(left_3(:,1).^2 + left_3(:,2).^2)'; %only the left side of the real movment
real_3 = horzcat(1,realright_3,realleft_3);%combining the two vectors into 1
real_3dis = abs(real_3(2:end)-2.5);  % the distance to the reak gauza circle while ignoring the first valur which add'd after horzcat 
realav_dis_3 = mean(real_3dis); % baseline average distance
disp(realav_dis_3)