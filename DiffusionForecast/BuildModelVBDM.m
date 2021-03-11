function [basis,ForwardOp,peq,qest,normalizer,delayEmbedding,epsilon] = BuildModelVBDM(x,M,k,k2,delays,shifts,dim)
%%% Inputs: x       - T x N time series of length T with dimension N
%%%         k       - number of nearest neighbors to use
%%%         k2      - nearest neighbors to use in pre-bandwidth
%%%         nvars   - number of eigenfunctions to find
%%%         delays  - (optional) delays to use, : delays = [1 5 20 50]
%%%         shifts  - (optional) number of shifts to use in the shift operator, default = 1
%%%         dim     - (optional) intrinsic dimension of the attractor

%%% Outputs: basis          - eigenfunctions of gradient flow on attractor
%%%          ForwardOp      - nvars x nvars forecast model in eigencoords.
%%%          peq            - invariant measure
%%%          qest           - sampling measure
%%%          normalizer     - tool to normalize coefficients of a density
%%%          delayEmbedding - delay embedded data
%%%          epsilon        - diffusion maps parameter (auto fit)

T=size(x,1);
N=size(x,2);

if (nargin < 3) k = ceil(sqrt(T));  end
if (nargin < 4) k2 = 16;            end
if (nargin < 5) delays = 1;         end
if (nargin < 6) shifts = 1;         end

L = length(delays);
maxd = max(delays);

delayEmbedding=zeros(T-maxd+1,N*L);

for i=1:L
    %%% Build the delay embedding of the time series
    delayEmbedding(:,(i-1)*N + (1:N)) = x(delays(i):T-maxd+delays(i),:);
end

operator = 2; %%% Build the generator for the gradient flow system

if (nargin < 7)
    [basis,~,epsilon,peq,qest] = VBDM(delayEmbedding,k,k2,M,operator);
else 
    [basis,~,epsilon,peq,qest] = VBDM(delayEmbedding,k,k2,M,operator,dim);
end

normalizer = mean(basis.*repmat(peq./qest,1,M));

T=T-maxd;

ForwardOp = (basis(1+shifts:T,:))'*(basis(1:T-shifts,:).*repmat(peq(1:T-shifts)./qest(1:T-shifts),1,M))/(T-shifts);







