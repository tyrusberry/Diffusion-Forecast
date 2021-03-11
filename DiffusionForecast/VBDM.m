function [q,b,epsilon,peq,qest] = VBDM(x,k,k2,nvars,operator,dim)

%%% Inputs
    %%% x       - N-by-n data set with N data points in R^n
    %%% k       - number of nearest neighbors to use
    %%% k2      - number of nearest neighbors to use to determine the "epsilon"
    %%%             parameter
    %%% nvars   - number of eigenfunctions/eigenvalues to compute
    %%% operator- 1 - Laplace-Beltrami operator, 2 - Kolmogorov backward operator 
    %%% dim     - intrinsic dimension of the manifold lying inside R^n
    %%% epsilon - optionally choose an arbitrary "global" epsilon
    
%%% Outputs
    %%% q       - Eigenfunctions of the generator/Laplacian
    %%% b       - Eigenvalues
    %%% epsilon - scale, derived from the k2 nearest neighbors
    %%% peqoversample - Invariant measure divided by the sampling measure
    %%% peq     - Invariant measure
    %%% qest    - Sampling measure

    
    %%% Theory requires c2 = 1/2 - 2*alpha + 2*dim*alpha + dim*beta/2 + beta < 0 
    %%% The resulting operator will have c1 = 2 - 2*alpha + dim*beta + 2*beta
    %%% Thus beta = (c1/2 - 1 + alpha)/(dim/2+1), since we want beta<0,
    %%% natural choices are beta=-1/2 or beta = -1/(dim/2+1)

    N = size(x,1); %number of points
    
    [d,inds] = knnCPU(x,x,k);

    %%% Build ad hoc bandwidth function by autotuning epsilon for each pt.
    
    epss = 2.^(-30:.1:10);

    rho0 = sqrt(mean(d(:,2:k2).^2,2));
    
    %%% Pre-kernel used with ad hoc bandwidth only for estimating dimension
    %%% and sampling density
    dt = d.^2./(repmat(rho0,1,k).*rho0(inds));
    
        %%% Tune epsilon on the pre-kernel
        dpreGlobal=zeros(1,length(epss));
        for i=1:length(epss)
            dpreGlobal(i) = sum(sum(exp(-dt./(2*epss(i)))))/(N*k);       
        end
        [maxval,maxind] = max(diff(log(dpreGlobal))./diff(log(epss)));
        if (nargin < 6)
            dim=2*maxval;
        end
    
    %%% Use ad hoc bandwidth function, rho0, to estimate the density
    dt = exp(-dt./(2*epss(maxind)))/((2*pi*epss(maxind))^(dim/2));
    dt = sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(dt'),N*k,1),N,N,N*k)';
    dt = (dt+dt')/2;

    %%% Kernel density estimate of the sampling density
    qest = (sum(dt,2))./(N*rho0.^(dim)); 
    
    clear dt;
    
    if (operator == 1)
        %%% Laplace-Beltrami, c1 = 0
        beta = -1/2;
        alpha = -dim/4 + 1/2;
    elseif (operator == 2)
        %%% Kolmogorov backward operator, c1 = 1
        beta = -1/2;
        alpha = -dim/4;     
    end

    c1 = 2 - 2*alpha + dim*beta + 2*beta;
    c2=.5-2*alpha+2*dim*alpha+dim*beta/2+beta;
    
    d = d.^2;
    
    %%% Define the true bandwidth function from the density estimate
    rho = qest.^(beta);
    rho = rho/mean(rho);

    d = d./repmat((rho),1,k);  % divide row j by rho(j)
    d = d./rho(inds);
    
        %%% Tune epsilon for the final kernel
        for i=1:length(epss)
            s(i) = sum(sum(exp(-d./(4*epss(i))),2))/(N*k);
        end
        [~,maxind] = max(diff(log(s))./diff(log(epss)));
        epsilon = epss(maxind);
   
    d = exp(-d./(4*epsilon));

    %%% Build the sparse kernel matrix
    d = sparse(reshape(double(inds'),N*k,1),repmat(1:N,k,1),reshape(double(d'),N*k,1),N,N,N*k)';
    clear inds;

    d = (d+d')/2;   %%% symmetrize since this is the symmetric formulation

    qest = full((sum(d,2)./(rho.^dim)));

    Dinv1 = spdiags(qest.^(-alpha),0,N,N);

    d = Dinv1*d*Dinv1; % the "right" normalization
    
    peqoversample = full((rho.^2).*(sum(d,2)));

    Dinv2 = spdiags(peqoversample.^(-1/2),0,N,N);

    d = Dinv2*d*Dinv2 - spdiags(rho.^(-2)-1,0,N,N); %%% "left" normalization

    opts.maxiter = 200;
    [q,b] = eigs(d,nvars,1,opts);
    
    q = Dinv2*q;
      
    %%% Sort by descending eigenvalues
    b = (diag(b));
    [~,perm] = sort(b,'descend');    
    b = b(perm).^(1/epsilon);
    b=diag(b);
    q = q(:,perm);

    %%% Normalize qest into a density by dividing by m0

    qest = qest/(N*(4*pi*epsilon)^(dim/2));
    peq = qest.*peqoversample;      %%% Invariant measure of the system
    peq = peq./mean(peq./qest);     %%% normalization factor

    %%% Normalize the eigenfunctions so their L^2 norm is 1
    %%% Note that the eigenfunctions are orthogonal with respect to
    %%% p_{eq}=qest^{c1} but sampled according to qest so we weight the
    %%% integrant by p_{eq}/qest = qest^{c1-1}.
 
    for i = 1:nvars
        q(:,i) = q(:,i)/sqrt(mean(q(:,i).^2.*(peq./qest)));
    end


    
end




