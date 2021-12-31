clc
clear all
close all
%M=csvread("WS_10m.csv",1,3);
%M=csvread("WS_hr.csv",2,3,[2 3 3000 3]);
M=csvread("WS_KFUPM_10m_2015.csv",1,2);
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%M=M(:,[2 4 5 6 7 8 10 

days=360;
%numdat=6*24*days;
inputsize=4;
%M=M(1:(numdat),:);
N=length(M);

for k=1:11
for i=1:N  
    if M(i,k)> 20 %CLEAN THE DATA if 9999, then replace with previous value
        M(i,k)=M(i-1,k);
    elseif M(i,k)<= 0
    M(i,k)=M(i-1,k);
    end
end
end

M=fliplr(M);

ii=1;
for i=1:N
    diff0=M(i,2:11)-M(i,1:10);
    lt0=sum(find(diff0<0.1));
    if lt0==0 
        MN(ii,:)=M(i,:);
        ii=ii+1;
    end
end

mt15=find(MN(:,6)<=15);   
M=MN(mt15,:);
mt10=find(M(:,3)<=10);   
M=M(mt10,:);

M=[M(:,[1 2 3 4]) (M(:,4)+M(:,5))/2 M(:,5) (M(:,5)+M(:,6))/2 M(:,6) (M(:,6)+M(:,7))/2 M(:,7) (M(:,7)+M(:,8))/2 M(:,8) (M(:,8)+M(:,9))/2 M(:,9) (M(:,9)+M(:,10))/2 M(:,10) (M(:,10)+M(:,11))/2 M(:,11)];
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%

perc=100;
numdat=length(M);

%R=6; % Every 6 makes an hour
%mm=floor(N/R);
%for i=1:mm
%    j=(i-1)*R+1;
%    MD(i,1)=mean(M(j:j+R-1));
%end

trainingnum=floor(0.8*numdat); % Num of training samples
maxx=max(max(M(1:trainingnum,1:inputsize)));
training=M(1:trainingnum,:);

series=training/maxx;
datasize=size(series);
nex=1;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing
Nhid=5;
Rrr=0.0000001;
testing=M((trainingnum+1):end,:);

seriesT=testing/maxx;
%numdata=max(datasize)-(inputsize+ahead-1);
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P50 = traininginput';
Y50 = trainingtarget';
Ptest50 = testinginput';
Ytest50 = testingtarget';
testingtarget50=Ytest50'*maxx;
%
%Create NN

%outval = netMLP(P);

trainingtargetmax=trainingtarget*maxx;

height50=[10 20 30 40 50];
rang50=[0 13];
rl50=[1:13];
% RELM WSE

RELMP50 = traininginput';
RELMY50 = trainingtarget';
RELMPtest50 = testinginput';
RELMYtest50 = testingtarget';
RELMtestingtarget50=RELMYtest50'*maxx;

Ww=10*randn(Nhid,inputsize+(nex-1));
%Hh=sigmoid(Ww*RELMP50);
Hh=(Ww*RELMP50);
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY50';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf50train=outvalmax';
%mse(RELMOutf50train,RELMY50*maxx)
%outvaltest=(sigmoid(Ww*RELMPtest50)'*Beta)';
outvaltest=((Ww*RELMPtest50)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget50;
RELMOutf50=outvaltestmax;
RELMmsetest50=mse(RELMOutf50,testingtarget50);
RELMmapetest50=mape(RELMOutf50,testingtarget50);
RELMmbetest50=mbe(RELMOutf50,testingtarget50);
RELMr2test50=rsquare(RELMOutf50,testingtarget50);
RELMperf50=[RELMmsetest50 RELMmapetest50 RELMmbetest50 RELMr2test50];
RELMPtestMax50=RELMPtest50'*maxx;

meantarget50=mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]');
meanRELM50=mean([RELMPtestMax50'; RELMOutf50' ]');
figure
plot(meantarget50,height50,'k');
hold on
plot(meanRELM50,height50,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf50]
perfall=[mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 60

nex=2;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y60 = trainingtarget';
Ytest60 = testingtarget';
testingtarget60=Ytest60'*maxx;

testingtargetmax=testingtarget*maxx;
target60=testingtarget60;

%
height60=[height50 60];
mxr=12.5+nex*0.5;
rang60=[0 mxr];
rl60=[1:mxr];
% RELM WSE
%
RELMP60 = [RELMP50; RELMOutf50train/maxx];
RELMY60 = trainingtarget';
RELMPtest60 = [RELMPtest50; RELMOutf50'/maxx];
RELMYtest60 = testingtarget';
RELMtestingtarget60=RELMYtest60'*maxx;


alpha=1/3;

Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP60;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY60';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf60train=outvalmax';
%mse(RELMOutf60train,RELMY60*maxx)
outvaltest=((Ww*RELMPtest60)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget60;
RELMOutf60=outvaltestmax;
RELMmsetest60=mse(RELMOutf60,testingtarget60);
RELMmapetest60=mape(RELMOutf60,testingtarget60);
RELMmbetest60=mbe(RELMOutf60,testingtarget60);
RELMr2test60=rsquare(RELMOutf60,testingtarget60);
RELMperf60=[RELMmsetest60 RELMmapetest60 RELMmbetest60 RELMr2test60];
RELMPtestMax60=RELMPtest60'*maxx;

meantarget60=[meantarget50 mean(testingtarget60)];
meanRELM60=[meanRELM50 mean(RELMOutf60)];

figure
plot(meantarget60,height60,'k');
hold on
plot(meanRELM60,height60,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf60]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 70

nex=3;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y70 = trainingtarget';
Ytest70 = testingtarget';
testingtarget70=Ytest70'*maxx;

testingtargetmax=testingtarget*maxx;
target70=testingtarget70;

%
height70=[height60 70];
mxr=12.5+nex*0.5;
rang70=[0 mxr];
rl70=[1:mxr];
% RELM WSE
%
RELMP70 = [RELMP60; RELMOutf60train/maxx];
RELMY70 = trainingtarget';
RELMPtest70 = [RELMPtest60; RELMOutf60'/maxx];
RELMYtest70 = testingtarget';
RELMtestingtarget70=RELMYtest70'*maxx;


Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP70;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY70';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf70train=outvalmax';
%mse(RELMOutf70train,RELMY70*maxx)
outvaltest=((Ww*RELMPtest70)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget70;
RELMOutf70=outvaltestmax;
RELMmsetest70=mse(RELMOutf70,testingtarget70);
RELMmapetest70=mape(RELMOutf70,testingtarget70);
RELMmbetest70=mbe(RELMOutf70,testingtarget70);
RELMr2test70=rsquare(RELMOutf70,testingtarget70);
RELMperf70=[RELMmsetest70 RELMmapetest70 RELMmbetest70 RELMr2test70];
RELMPtestMax70=RELMPtest70'*maxx;

meantarget70=[meantarget60 mean(testingtarget70)];
meanRELM70=[meanRELM60 mean(RELMOutf70)];

figure
plot(meantarget70,height70,'k');
hold on
plot(meanRELM70,height70,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf70]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 80

nex=4;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y80 = trainingtarget';
Ytest80 = testingtarget';
testingtarget80=Ytest80'*maxx;

testingtargetmax=testingtarget*maxx;
target80=testingtarget80;

%
height80=[height70 80];
mxr=12.5+nex*0.5;
rang80=[0 mxr];
rl80=[1:mxr];
% RELM WSE
%
RELMP80 = [RELMP70; RELMOutf70train/maxx];
RELMY80 = trainingtarget';
RELMPtest80 = [RELMPtest70; RELMOutf70'/maxx];
RELMYtest80 = testingtarget';
RELMtestingtarget80=RELMYtest80'*maxx;

Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP80;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY80';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf80train=outvalmax';
%mse(RELMOutf80train,RELMY80*maxx)
outvaltest=((Ww*RELMPtest80)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget80;
RELMOutf80=outvaltestmax;
RELMmsetest80=mse(RELMOutf80,testingtarget80);
RELMmapetest80=mape(RELMOutf80,testingtarget80);
RELMmbetest80=mbe(RELMOutf80,testingtarget80);
RELMr2test80=rsquare(RELMOutf80,testingtarget80);
RELMperf80=[RELMmsetest80 RELMmapetest80 RELMmbetest80 RELMr2test80];
RELMPtestMax80=RELMPtest80'*maxx;

meantarget80=[meantarget70 mean(testingtarget80)];
meanRELM80=[meanRELM70 mean(RELMOutf80)];

figure
plot(meantarget80,height80,'k');
hold on
plot(meanRELM80,height80,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('80 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf80]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 90

nex=5;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y90 = trainingtarget';
Ytest90 = testingtarget';
testingtarget90=Ytest90'*maxx;

testingtargetmax=testingtarget*maxx;
target90=testingtarget90;

%
height90=[height80 90];
mxr=12.5+nex*0.5;
rang90=[0 mxr];
rl90=[1:mxr];
% RELM WSE
%
RELMP90 = [RELMP80; RELMOutf80train/maxx];
RELMY90 = trainingtarget';
RELMPtest90 = [RELMPtest80; RELMOutf80'/maxx];
RELMYtest90 = testingtarget';
RELMtestingtarget90=RELMYtest90'*maxx;


Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP90;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY90';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf90train=outvalmax';
%mse(RELMOutf90train,RELMY90*maxx)
outvaltest=((Ww*RELMPtest90)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget90;
RELMOutf90=outvaltestmax;
RELMmsetest90=mse(RELMOutf90,testingtarget90);
RELMmapetest90=mape(RELMOutf90,testingtarget90);
RELMmbetest90=mbe(RELMOutf90,testingtarget90);
RELMr2test90=rsquare(RELMOutf90,testingtarget90);
RELMperf90=[RELMmsetest90 RELMmapetest90 RELMmbetest90 RELMr2test90];
RELMPtestMax90=RELMPtest90'*maxx;

meantarget90=[meantarget80 mean(testingtarget90)];
meanRELM90=[meanRELM80 mean(RELMOutf90)];

figure
plot(meantarget90,height90,'k');
hold on
plot(meanRELM90,height90,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('90 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf90]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 100

nex=6;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y100 = trainingtarget';
Ytest100 = testingtarget';
testingtarget100=Ytest100'*maxx;

testingtargetmax=testingtarget*maxx;
target100=testingtarget100;

%
height100=[height90 100];
mxr=12.5+nex*0.5;
rang100=[0 mxr];
rl100=[1:mxr];
% RELM WSE
%
RELMP100 = [RELMP90; RELMOutf90train/maxx];
RELMY100 = trainingtarget';
RELMPtest100 = [RELMPtest90; RELMOutf90'/maxx];
RELMYtest100 = testingtarget';
RELMtestingtarget100=RELMYtest100'*maxx;

Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP100;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY100';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf100train=outvalmax';
%mse(RELMOutf100train,RELMY100*maxx)
outvaltest=((Ww*RELMPtest100)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget100;
RELMOutf100=outvaltestmax;
RELMmsetest100=mse(RELMOutf100,testingtarget100);
RELMmapetest100=mape(RELMOutf100,testingtarget100);
RELMmbetest100=mbe(RELMOutf100,testingtarget100);
RELMr2test100=rsquare(RELMOutf100,testingtarget100);
RELMperf100=[RELMmsetest100 RELMmapetest100 RELMmbetest100 RELMr2test100];
RELMPtestMax100=RELMPtest100'*maxx;

meantarget100=[meantarget90 mean(testingtarget100)];
meanRELM100=[meanRELM90 mean(RELMOutf100)];

figure
plot(meantarget100,height100,'k');
hold on
plot(meanRELM100,height100,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('100 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf100]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 110

nex=7;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y110 = trainingtarget';
Ytest110 = testingtarget';
testingtarget110=Ytest110'*maxx;

testingtargetmax=testingtarget*maxx;
target110=testingtarget110;

%
height110=[height100 110];
mxr=12.5+nex*0.5;
rang110=[0 mxr];
rl110=[1:mxr];
% RELM WSE
%
RELMP110 = [RELMP100; RELMOutf100train/maxx];
RELMY110 = trainingtarget';
RELMPtest110 = [RELMPtest100; RELMOutf100'/maxx];
RELMYtest110 = testingtarget';
RELMtestingtarget110=RELMYtest110'*maxx;


Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP110;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY110';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf110train=outvalmax';
%mse(RELMOutf110train,RELMY110*maxx)
outvaltest=((Ww*RELMPtest110)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget110;
RELMOutf110=outvaltestmax;
RELMmsetest110=mse(RELMOutf110,testingtarget110);
RELMmapetest110=mape(RELMOutf110,testingtarget110);
RELMmbetest110=mbe(RELMOutf110,testingtarget110);
RELMr2test110=rsquare(RELMOutf110,testingtarget110);
RELMperf110=[RELMmsetest110 RELMmapetest110 RELMmbetest110 RELMr2test110];
RELMPtestMax110=RELMPtest110'*maxx;

meantarget110=[meantarget100 mean(testingtarget110)];
meanRELM110=[meanRELM100 mean(RELMOutf110)];

figure
plot(meantarget110,height110,'k');
hold on
plot(meanRELM110,height110,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('110 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf110]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 120

nex=8;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y120 = trainingtarget';
Ytest120 = testingtarget';
testingtarget120=Ytest120'*maxx;

testingtargetmax=testingtarget*maxx;
target120=testingtarget120;

%
height120=[height110 120];
mxr=12.5+nex*0.5;
rang120=[0 mxr];
rl120=[1:mxr];
% RELM WSE
%
RELMP120 = [RELMP110; RELMOutf110train/maxx];
RELMY120 = trainingtarget';
RELMPtest120 = [RELMPtest110; RELMOutf110'/maxx];
RELMYtest120 = testingtarget';
RELMtestingtarget120=RELMYtest120'*maxx;

Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP120;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY120';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf120train=outvalmax';
%mse(RELMOutf120train,RELMY120*maxx)
outvaltest=((Ww*RELMPtest120)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget120;
RELMOutf120=outvaltestmax;
RELMmsetest120=mse(RELMOutf120,testingtarget120);
RELMmapetest120=mape(RELMOutf120,testingtarget120);
RELMmbetest120=mbe(RELMOutf120,testingtarget120);
RELMr2test120=rsquare(RELMOutf120,testingtarget120);
RELMperf120=[RELMmsetest120 RELMmapetest120 RELMmbetest120 RELMr2test120];
RELMPtestMax120=RELMPtest120'*maxx;

meantarget120=[meantarget110 mean(testingtarget120)];
meanRELM120=[meanRELM110 mean(RELMOutf120)];

figure
plot(meantarget120,height120,'k');
hold on
plot(meanRELM120,height120,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('120 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf120]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 130

nex=9;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y130 = trainingtarget';
Ytest130 = testingtarget';
testingtarget130=Ytest130'*maxx;

testingtargetmax=testingtarget*maxx;
target130=testingtarget130;

%
height130=[height120 130];
mxr=12.5+nex*0.5;
rang130=[0 mxr];
rl130=[1:mxr];
% RELM WSE
%
RELMP130 = [RELMP120; RELMOutf120train/maxx];
RELMY130 = trainingtarget';
RELMPtest130 = [RELMPtest120; RELMOutf120'/maxx];
RELMYtest130 = testingtarget';
RELMtestingtarget130=RELMYtest130'*maxx;

Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP130;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY130';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf130train=outvalmax';
%mse(RELMOutf130train,RELMY130*maxx)
outvaltest=((Ww*RELMPtest130)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget130;
RELMOutf130=outvaltestmax;
RELMmsetest130=mse(RELMOutf130,testingtarget130);
RELMmapetest130=mape(RELMOutf130,testingtarget130);
RELMmbetest130=mbe(RELMOutf130,testingtarget130);
RELMr2test130=rsquare(RELMOutf130,testingtarget130);
RELMperf130=[RELMmsetest130 RELMmapetest130 RELMmbetest130 RELMr2test130];
RELMPtestMax130=RELMPtest130'*maxx;

meantarget130=[meantarget120 mean(testingtarget130)];
meanRELM130=[meanRELM120 mean(RELMOutf130)];

figure
plot(meantarget130,height130,'k');
hold on
plot(meanRELM130,height130,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('130 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf130]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 140

nex=10;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y140 = trainingtarget';
Ytest140 = testingtarget';
testingtarget140=Ytest140'*maxx;

testingtargetmax=testingtarget*maxx;
target140=testingtarget140;

%
height140=[height130 140];
mxr=12.5+nex*0.5;
rang140=[0 mxr];
rl140=[1:mxr];
% RELM WSE
%
RELMP140 = [RELMP130; RELMOutf130train/maxx];
RELMY140 = trainingtarget';
RELMPtest140 = [RELMPtest130; RELMOutf130'/maxx];
RELMYtest140 = testingtarget';
RELMtestingtarget140=RELMYtest140'*maxx;

Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP140;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY140';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf140train=outvalmax';
%mse(RELMOutf140train,RELMY140*maxx)
outvaltest=((Ww*RELMPtest140)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget140;
RELMOutf140=outvaltestmax;
RELMmsetest140=mse(RELMOutf140,testingtarget140);
RELMmapetest140=mape(RELMOutf140,testingtarget140);
RELMmbetest140=mbe(RELMOutf140,testingtarget140);
RELMr2test140=rsquare(RELMOutf140,testingtarget140);
RELMperf140=[RELMmsetest140 RELMmapetest140 RELMmbetest140 RELMr2test140];
RELMPtestMax140=RELMPtest140'*maxx;

meantarget140=[meantarget130 mean(testingtarget140)];
meanRELM140=[meanRELM130 mean(RELMOutf140)];

figure
plot(meantarget140,height140,'k');
hold on
plot(meanRELM140,height140,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('140 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf140]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 150

nex=11;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y150 = trainingtarget';
Ytest150 = testingtarget';
testingtarget150=Ytest150'*maxx;

testingtargetmax=testingtarget*maxx;
target150=testingtarget150;

%
height150=[height140 150];
mxr=12.5+nex*0.5;
rang150=[0 mxr];
rl150=[1:mxr];
% RELM WSE
%
RELMP150 = [RELMP140; RELMOutf140train/maxx];
RELMY150 = trainingtarget';
RELMPtest150 = [RELMPtest140; RELMOutf140'/maxx];
RELMYtest150 = testingtarget';
RELMtestingtarget150=RELMYtest150'*maxx;


Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP150;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY150';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf150train=outvalmax';
%mse(RELMOutf150train,RELMY150*maxx)
outvaltest=((Ww*RELMPtest150)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget150;
RELMOutf150=outvaltestmax;
RELMmsetest150=mse(RELMOutf150,testingtarget150);
RELMmapetest150=mape(RELMOutf150,testingtarget150);
RELMmbetest150=mbe(RELMOutf150,testingtarget150);
RELMr2test150=rsquare(RELMOutf150,testingtarget150);
RELMperf150=[RELMmsetest150 RELMmapetest150 RELMmbetest150 RELMr2test150];
RELMPtestMax150=RELMPtest150'*maxx;

meantarget150=[meantarget140 mean(testingtarget150)];
meanRELM150=[meanRELM140 mean(RELMOutf150)];

figure
plot(meantarget150,height150,'k');
hold on
plot(meanRELM150,height150,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('150 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf150]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 160

nex=12;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y160 = trainingtarget';
Ytest160 = testingtarget';
testingtarget160=Ytest160'*maxx;

testingtargetmax=testingtarget*maxx;
target160=testingtarget160;

%
height160=[height150 160];
mxr=12.5+nex*0.5;
rang160=[0 mxr];
rl160=[1:mxr];
% RELM WSE
%
RELMP160 = [RELMP150; RELMOutf150train/maxx];
RELMY160 = trainingtarget';
RELMPtest160 = [RELMPtest150; RELMOutf150'/maxx];
RELMYtest160 = testingtarget';
RELMtestingtarget160=RELMYtest160'*maxx;


Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP160;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY160';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf160train=outvalmax';
%mse(RELMOutf160train,RELMY160*maxx)
outvaltest=((Ww*RELMPtest160)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget160;
RELMOutf160=outvaltestmax;
RELMmsetest160=mse(RELMOutf160,testingtarget160);
RELMmapetest160=mape(RELMOutf160,testingtarget160);
RELMmbetest160=mbe(RELMOutf160,testingtarget160);
RELMr2test160=rsquare(RELMOutf160,testingtarget160);
RELMperf160=[RELMmsetest160 RELMmapetest160 RELMmbetest160 RELMr2test160];
RELMPtestMax160=RELMPtest160'*maxx;

meantarget160=[meantarget150 mean(testingtarget160)];
meanRELM160=[meanRELM150 mean(RELMOutf160)];

figure
plot(meantarget160,height160,'k');
hold on
plot(meanRELM160,height160,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('160 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf160]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 170

nex=13;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y170 = trainingtarget';
Ytest170 = testingtarget';
testingtarget170=Ytest170'*maxx;

testingtargetmax=testingtarget*maxx;
target170=testingtarget170;

%
height170=[height160 170];
mxr=12.5+nex*0.5;
rang170=[0 mxr];
rl170=[1:mxr];
% RELM WSE
%
RELMP170 = [RELMP160; RELMOutf160train/maxx];
RELMY170 = trainingtarget';
RELMPtest170 = [RELMPtest160; RELMOutf160'/maxx];
RELMYtest170 = testingtarget';
RELMtestingtarget170=RELMYtest170'*maxx;


Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP170;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY170';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf170train=outvalmax';
%mse(RELMOutf170train,RELMY170*maxx)
outvaltest=((Ww*RELMPtest170)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget170;
RELMOutf170=outvaltestmax;
RELMmsetest170=mse(RELMOutf170,testingtarget170);
RELMmapetest170=mape(RELMOutf170,testingtarget170);
RELMmbetest170=mbe(RELMOutf170,testingtarget170);
RELMr2test170=rsquare(RELMOutf170,testingtarget170);
RELMperf170=[RELMmsetest170 RELMmapetest170 RELMmbetest170 RELMr2test170];
RELMPtestMax170=RELMPtest170'*maxx;

meantarget170=[meantarget160 mean(testingtarget170)];
meanRELM170=[meanRELM160 mean(RELMOutf170)];

figure
plot(meantarget170,height170,'k');
hold on
plot(meanRELM170,height170,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('170 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf170]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 180

nex=14;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y180 = trainingtarget';
Ytest180 = testingtarget';
testingtarget180=Ytest180'*maxx;

testingtargetmax=testingtarget*maxx;
target180=testingtarget180;

%
height180=[height170 180];
mxr=12.5+nex*0.5;
rang180=[0 mxr];
rl180=[1:mxr];
% RELM WSE
%
RELMP180 = [RELMP170; RELMOutf170train/maxx];
RELMY180 = trainingtarget';
RELMPtest180 = [RELMPtest170; RELMOutf170'/maxx];
RELMYtest180 = testingtarget';
RELMtestingtarget180=RELMYtest180'*maxx;

Ww=10*randn(Nhid,inputsize+(nex-1));
Hh=Ww*RELMP180;
Beta=inv(1/Rrr+Hh*Hh')*Hh*RELMY180';

outval=Hh'*Beta;
outvalmax=outval*maxx;
RELMOutf180train=outvalmax';
%mse(RELMOutf180train,RELMY180*maxx)
outvaltest=((Ww*RELMPtest180)'*Beta)';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget180;
RELMOutf180=outvaltestmax;
RELMmsetest180=mse(RELMOutf180,testingtarget180);
RELMmapetest180=mape(RELMOutf180,testingtarget180);
RELMmbetest180=mbe(RELMOutf180,testingtarget180);
RELMr2test180=rsquare(RELMOutf180,testingtarget180);
RELMperf180=[RELMmsetest180 RELMmapetest180 RELMmbetest180 RELMr2test180];
RELMPtestMax180=RELMPtest180'*maxx;

meantarget180=[meantarget170 mean(testingtarget180)];
meanRELM180=[meanRELM170 mean(RELMOutf180)];

figure
plot(meantarget180,height180,'k');
hold on
plot(meanRELM180,height180,'-.g');

hold off
title('average')
legend('measured','RELM est','Location','northwest')

display ('180 m Testing:     MSE     MAPE        MBE        R2')
[RELMperf180]
perfall=[perfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
ELMperfall=perfall;