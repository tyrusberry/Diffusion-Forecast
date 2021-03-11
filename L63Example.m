clear;clc;close all;
addpath('ExampleDynamics');
addpath('DiffusionForecast');

%%%% PLEASE NOTE: THIS IS A LONG RUN VERIFICATION PROGRAM AND IT REQUIRES
%%%% APPROXIMATELY 2GB OF RAM AND ABOUT 90 MINUTES RUNTIME ON A 2013 MacBookPro

%%%%%%%%%%%%%%%%%%%%%% Load Data %%%%%%%%%%%%%%%%%%%%%%

tau = .1;               %%% Discrete time step
ftime = 10;             %%% Number of model time units to forecast
fsteps=ceil(ftime/tau); %%% Number of forecast steps       

N = 10500;      %%% Number of data points to generate
T = tau*N;      %%% Total time of simulation
b = 0;          %%% Use the purely deterministic L63, no stochastic forcing

%%% Choose a random initial condition and run the dynamics to eliminate any transient
x0 = rand(3,1);
[x0,~] = L63(x0,100,tau,b);
x0 = x0(end,:)';

%%% Using the initial condition x0 which is now on the manifold we produce the data set
[x,t] = L63(x0,T,tau,b);

%%% Train the nonparametric model using the first 5000 data points
trainingData = x(1:5000,:);

verificationData = x(5001:end,:);

N = size(trainingData,1);
n = size(trainingData,2);

Tplot = size(verificationData,1);


%%%%%%%%%%%%%%%%%%%%%% Build Nonparametric Model %%%%%%%%%%%%%%%%%%%%%%

k=512;      %%% Number of nearest neighbors to use in building the basis
k2=64;      %%% Number of nearest neighbors to use in the density estimate
M=4000;     %%% Number of basis functions to build

[basis,ForwardOp,peq,qest,normalizer] = BuildModelVBDM(trainingData,M,k,k2);



%%%%%%%%%%%%%%%%%%%%%% Out of Sample Forecast %%%%%%%%%%%%%%%%%%%%%%


%%% Functionals which we will compute the expectation of are projected into the basis
meanCoeff = trainingData'*(basis.*repmat(peq./qest,1,M))/N;
varCoeff = (trainingData.^2)'*(basis.*repmat(peq./qest,1,M))/N;

verificationLength = 500; %%% We use 5000 in the paper which requires about 10 hours to run

forecastErr = zeros(fsteps,n);
fvar = zeros(fsteps,n);
llferr = zeros(fsteps,n);
llfvar = zeros(fsteps,n);
ensferr = zeros(fsteps,n);
ensfvar = zeros(fsteps,n);
ll2ferr = zeros(fsteps,n);
ll2fvar = zeros(fsteps,n);

for i = 1:verificationLength
    
    perturbationVar = .01;

    %%% Build the initial condition by a small random perturbation of the true verification point    
    currState = verificationData(i,:); 
    currState = currState + sqrt(perturbationVar)*randn(size(currState));

    %%% Construct Nonparametric Initial Condition
    ds = sum((trainingData - repmat(currState,N,1)).^2,2);
    ptemp = exp(-ds/(4*perturbationVar));
    
    ctemp = basis'*(ptemp./qest)/N;
    ctemp = ctemp/(normalizer*ctemp);   
    
    %%% Construct Ensemble Initial Condition
    Pb = .01*eye(3);    %%% Covariance of the random perturbation
    rootPb = .1*eye(3); %%% Square root of the perturbation
    ens = repmat(currState',1,50000) + rootPb*randn(3,50000);
    
    %%% Nearest neighbor indices for local linear forecast
    
    [dstemp,indstemp]=sort(ds);
    indstemp = indstemp(1:15);
    
    for j=1:fsteps
        
            truth = verificationData(i+j-1,:);
        
            %%% Nonparametric forecast

            Forecast = meanCoeff*ctemp;
            forecastErr(j,:) = forecastErr(j,:) + (truth - Forecast').^2/verificationLength;

            ForecastVar = sqrt(abs(varCoeff*ctemp - Forecast.^2));
            fvar(j,:) = fvar(j,:) + ForecastVar'/verificationLength;  

            ctemp = ForwardOp*ctemp;
            ctemp = ctemp/(normalizer*ctemp);
            
            
            %%% Direct local linear forecast
            
            nn = trainingData(indstemp(indstemp<size(trainingData,1)-j+1),:);
            Fnn = trainingData(indstemp(indstemp<size(trainingData,1)-j+1)+j-1,:);
            munn = mean(nn);
            muFnn = mean(Fnn);
            llModel = (nn - repmat(munn,size(nn,1),1))\(Fnn-repmat(muFnn,size(Fnn,1),1));
            tf = (currState - munn)*llModel + muFnn;
            llForecast = tf;
            tv = sqrt(abs(diag(llModel'*Pb*llModel)));
            llForecastVar = tv';
            llferr(j,:) = llferr(j,:) + (truth - llForecast).^2/verificationLength;        
            llfvar(j,:) = llfvar(j,:) + llForecastVar/verificationLength;
            
            
            %%% Ensemble forecast
            
            ensForecast = mean(ens');
            ensForecastVar = std(ens');
            
            ensferr(j,:) = ensferr(j,:) + (truth - ensForecast).^2/verificationLength;
            ensfvar(j,:) = ensfvar(j,:) + ensForecastVar/verificationLength; 
            
            ens = L63(ens,tau,tau,b);
            ens = squeeze(ens(end,:,:));
            
    end

    
    %%% Iterated local linear forecast
    
    ll2Forecast = currState;
    currStateLL = currState;
    
    for j=1:fsteps

            truth = verificationData(i+j-1,:);
            tv = sqrt(abs(diag(Pb)));
            ll2ForecastVar = tv';
            ll2ferr(j,:) = ll2ferr(j,:) + (truth - ll2Forecast).^2/verificationLength;        
            ll2fvar(j,:) = ll2fvar(j,:) + ll2ForecastVar/verificationLength;
            ds = sum((trainingData - repmat(currStateLL,N,1)).^2,2);
            [dstemp,indstemp]=sort(ds);
            indstemp = indstemp(1:15);
            nn = trainingData(indstemp(indstemp<size(trainingData,1)-1),:);
            Fnn = trainingData(indstemp(indstemp<size(trainingData,1)-1)+1,:);
            munn = mean(nn);
            muFnn = mean(Fnn);
            llModel = (nn - repmat(munn,size(nn,1),1))\(Fnn-repmat(muFnn,size(Fnn,1),1));
            tf = (currStateLL - munn)*llModel + muFnn;
            currStateLL = tf;
            if (j<fsteps)
                ll2Forecast = tf;
                Pb = llModel'*Pb*llModel;
            end
            
    end

    
end

forecastErr = sqrt(forecastErr);
llferr = sqrt(llferr);
ensferr = sqrt(ensferr);
ll2ferr = sqrt(ll2ferr);



%%%%%%%%%%%%%%%%%%%%%% Plot %%%%%%%%%%%%%%%%%%%%%%


plotInd=1;
figure(12);hold off;
t=tau*(0:fsteps-1);
semilogy(t,ensferr(:,plotInd),'color',[.7 .7 .7],'linewidth',3);hold on;
semilogy(t,ensfvar(:,plotInd),'--','color',[.6 .6 .6],'linewidth',3);
semilogy(t,forecastErr(:,plotInd),'linewidth',2);hold on;
semilogy(t,fvar(:,plotInd),'--','linewidth',2);
semilogy(t,llferr(:,plotInd),'r','linewidth',1.5);
semilogy(t,llfvar(:,plotInd),'r--','linewidth',1.5);
semilogy(t,ll2ferr(:,plotInd),'k','linewidth',1);
semilogy(t,ll2fvar(:,plotInd),'k--','linewidth',1);
semilogy(t,ones(fsteps,1)*std(trainingData(:,plotInd)),'k:');
l=legend('Ensemble Forecast','Ensemble Error Estimate','Diffusion Forecast','Diffusion Error Estimate','Local Linear Forecast','Local Linear Error Estimate','Iterated Local Linear Forecast','Iterated Local Linear Error Estimate','Invariant Measure',4);
set(l,'FontSize',12);
xlabel('Forecast Time','FontSize',14);
ylabel('RMSE','FontSize',14);
ylim([0 100]);
xlim([0 10.5]);
set(gca,'FontSize',14);


