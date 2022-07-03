function y= Balda_QRSs(x,threshold,win_length,Fs,flag)
%BALDA_QRSS takes the ECG signal, window length, threshold and sampling
%frequency and finds QRS comlex by two integrations. the function also
%provide the option to plot (if flag=1)


x = x - mean (x );    % cancel DC components
x = x/ max( abs(x )); % normalize to one


%first order derivative in time domain
first_div=[1 0 -1];
y_0=conv(x,first_div);

%second order derivative in time domain
second_div=[1 0 -2 0 1];
y_1=conv(x,second_div);
% add the two derivatives
y_2=1.1*abs(y_1(3:end-2))+1.3*abs(y_0(2:end-1));
Y=y_2;

win=[ones(1,win_length) ];
% now we want to detect when we have win_len-2 successes above the
% threshold. to do so we created a window of 1 in length on Win_len. after 
% the convolution, we ask in a logical way what the product is 
detection_win_len_pks_in_row=conv(Y>=threshold,win)>=win_length-2;

y=cat(1,zeros(1,win_length/2)',detection_win_len_pks_in_row(win_length:end-win_length/2)');

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

