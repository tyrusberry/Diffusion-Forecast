clear;clc;close all;

addpath('ExampleDynamics');
addpath('DiffusionForecast');

%%%% PLEASE NOTE: THIS IS A LONG RUN VERIFICATION PROGRAM AND IT REQUIRES
%%%% APPROXIMATELY 4GB OF RAM AND ABOUT 45 MINUTES RUNTIME ON A 2013 MacBookPro

%%%%%%%%%%%%%%%%%%%%%% Generate Training Data %%%%%%%%%%%%%%%%%%%%%%

N = 20000;  %%% Number of data points
tau = .1;   %%% Discrete time step
T = N*tau;  %%% Total time to simulate   

%%% Initial Conditions in the intrinsic coordinate (theta,phi) in [0,2pi)^2
theta0 = 1;
phi0 = 1;

%%% Diffusion coefficient "b" for the stochastic dynamical system
b = 1;

%%% Produce a training data set
[x,y,z,t,theta,phi] = torusSDE(theta0,phi0,T,tau,b);

%%% We are given data on a torus in R^3
trainingData = [x y z]; 



%%%%%%%%%%%%%%%%%%%%%% Build Nonparametric Model %%%%%%%%%%%%%%%%%%%%%%

k=1024;     %%% Number of nearest neighbors to use in building the basis
k2=64;      %%% Number of nearest neighbors to use in the density estimate
M=1000;     %%% Number of basis functions to build

[basis,ForwardOp,peq,qest,normalizer] = BuildModelVBDM(trainingData,M,k,k2);



%%%%%%%%%%%%%%%%%%%%%% Forecast Comparison %%%%%%%%%%%%%%%%%%%%%%

%%% AFTER CODE HAS RUN YOU CAN COPY-PASTE THIS SECTION TO VIEW THE EVOLUTION

%%% Construct an ensemble of 50000 points sampled from a Gaussian initial density
ivar = 1/10;
ensSize = 50000;
xEns = mod(repmat([theta0;phi0],1,ensSize) + sqrt(ivar)*randn(2,ensSize),2*pi);

%%% Construct the same Gaussian initial density by evaluating on the training data set
p0 = exp(-(theta-theta0).^2/2/ivar - (phi-phi0).^2/2/ivar)/(2*pi*ivar);

%%% Project the initial density into the basis by computing the coefficients 
c = (basis)'*(p0./qest)/N;

%%% Functionals which we will compute the expectation of are projected into the basis
meanCoeff = trainingData'*(basis.*repmat(peq./qest,1,M))/N;
varCoeff = (trainingData.^2)'*(basis.*repmat(peq./qest,1,M))/N;


figure(1);set(gcf,'position',[300 50 800 300]);

%%% Compare Monte-Carlo to Diffusion forecast
for i = 1:300
    
    %%% Diffusion Forecast
    c = ForwardOp*c;        %%% Push forward the coefficients of the density
    c = c/(normalizer*c);   %%% Renormalize the density coefficients
    
    estMean(i,:) = meanCoeff*c; %%% Compute the mean of the density
    estVar(i,:) = (abs(varCoeff*c - estMean(i,:)'.^2)); %%% compute the variance
    
    recon = peq.*(basis*c); %%% Reconstruct the density for plotting
    
    %%% Ensemble Forecast using the true dynamics
    [xe,ye,ze,~,xx,yy] = torusSDE(xEns(1,:),xEns(2,:),2*tau,tau,b);
    xEns(1,:) = xx(end,:);
    xEns(2,:) = yy(end,:);

    xe=xe(2,:);ye=ye(2,:);ze=ze(2,:);
    trueMean(i,:) = mean([xe;ye;ze]');
    trueVar(i,:) = std([xe;ye;ze]').^2;
  
    
        subplot(1,2,1); %%% Draw diffusion forecast, 
        scatter(theta,phi,5,(2+sin(theta)).*recon,'filled');
        %%% The factor (2+sin(theta)) is the volume form of the torus
        %%% (since we are plotting the density in intrinsic coordinates)
        xlim([0 2*pi]);ylim([0 2*pi]);set(gca,'FontSize',14);
        xlabel('\theta','FontSize',14);ylabel('\phi','FontSize',14);
        title('Nonparametric Diffusion Forecast','FontSize',14);
        
        subplot(1,2,2);
        plot(xEns(1,1:500),xEns(2,1:500),'.')
        xlim([0 2*pi]);ylim([0 2*pi]);set(gca,'FontSize',14);
        xlabel('\theta','FontSize',14);ylabel('\phi','FontSize',14);
        title('Ensemble Forecast with True Model','FontSize',14);

        drawnow;

end



%%%%%%%%%%%%%%%%%%%%%% Comparison of Forecast Moments %%%%%%%%%%%%%%%%%%%%%%

figure(2);hold off;
plot(trueMean(:,1),'color',[.3 .3 .3],'linewidth',3);hold on;
plot(estMean(:,1),'r','linewidth',1.5);
plot(estVar(:,1),'r:','linewidth',1.5);
plot(trueMean(:,3),'color',[.7 .7 .7],'linewidth',4);hold on;
plot(estMean(:,3),'b','linewidth',2);
plot(estVar(:,3),'b--','linewidth',2);

plot(trueMean(:,1),'color',[.3 .3 .3],'linewidth',3);hold on;
plot(trueVar(:,1),'color',[.3 .3 .3],'linewidth',3);
plot(estMean(:,1),'r','linewidth',1.5);
plot(estVar(:,1),'r:','linewidth',1.5);

plot(trueMean(:,3),'color',[.7 .7 .7],'linewidth',4);hold on;
plot(trueVar(:,3),'color',[.7 .7 .7],'linewidth',4);
plot(estMean(:,3),'b','linewidth',2);
plot(estVar(:,3),'b--','linewidth',2);

l=legend('Ensemble forecast, mean and variance of x','Diffusion forecast, mean of x','Diffusion forecast, variance of x','Ensemble forecast, mean and variance of z','Diffusion forecast, mean of z','Diffusion forecast, variance of z',4);
set(l,'FontSize',14);
xlabel('Forecast Steps','FontSize',14);
ylabel('Forecast Value','FontSize',14);
xlim([1 250]);
ylim([-2.7 3.1]);
set(gca,'FontSize',14);


