function y = PanThomp_QRSs(x,Win_len,Fs,threshold,flag)
%PanThomp_QRS takes the ECG signal', window length, threshold and sampling
%frequency and finds QRS comlex by two integrations
%flag give the option to plot
N = length (x);      


%CANCELLATION DC DRIFT AND NORMALIZATION
x = x - mean (x );    % cancel DC components
x = x/ max( abs(x )); % normalize to one

%First, in order to attenuate noise, the signal passes through a
%digital bandpass filter composed of cascaded high-pass and lowpass filters.


% LPF (1-z^-6)^2/(1-z^-1)^2
b=[1 0 0 0 0 0 -2 0 0 0 0 0 1]*1/32;
a=[1 -2 1];
h_LP=filter(b,a,[1 zeros(1,12)]); % transfer function of LPF
x2 = conv (x ,h_LP);
x2 = x2(7:end-6)/ max( abs(x2(7:end-6) )); 


% HPF = Allpass-(Lowpass) = z^-16-[(1-z^-32)/(1-z^-1)]
b = [-1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 -32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]*1/32;
a = [1 -1];
h_HP=filter(b,a,[1 zeros(1,32)]); % impulse response of HPF
x3 = conv (x2 ,h_HP);
x3 = x3(17:end-16)/ max( abs(x3(17:end-16) ));


% Make impulse response
h = [2 1 0 -1 -2]/8;
x4 = conv (x3 ,h);
x4 = x4(3:end-2)/ max( abs(x4(3:end-2) ));

% squaring
x5 = x4 .^2;
x5 = x5/ max( abs(x5 ));

% moving average integrator
N=30;
h_Ma=ones(1,N)*1/N ;

x6=conv(x5,h_Ma);

x6=x6(N/2:end-N/2);
x6=x6/max(abs(x6));
Y=x6;

% mooving average window
win=[ones(1,Win_len) ];
% now we want to detect when we have win_len-2 successes above the
% threshold. to do so we created a window of 1 in length on Win_len. after 
% the convolution, we ask in a logical way whe the product is 
detection_win_len_pks_in_row=conv(Y>=threshold,win)>=Win_len-2;
y=[zeros(1,Win_len/2) detection_win_len_pks_in_row(Win_len:end-Win_len/2)];

y=islocalmax(y);
t= linspace(0,length(Y),length(Y))*(1/Fs);

if flag==1
    
figure
subplot(211)
plot(t,Y)
  xlim([0 3]); title('Y_2 - Weighted Sum  '); ylabel('amp [mV]'); grid on
 hold on
yline(threshold,'color','r','LineWidth',1.2);legend('y_2',['threshold=' num2str(threshold) ''],'location','northwest')
 
  subplot(212)
plot(linspace(0,length(x),length(x))*(1/Fs),x)
  xlim([0 3]); title('Output of Balda QRS function'); ylabel('amp [mV]'); xlabel('time [Sec]'); grid on
 hold on
plot(t,y); legend('ECG','QRS','location','northwest')
end
end

