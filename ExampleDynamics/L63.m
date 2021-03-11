function [x,t] = L63(x0,T,tau,D)

    t = 0:tau:T;
    N = length(t);

    x = zeros(N,size(x0,1),size(x0,2));

    x(1,:,:) = x0;
    state = x0;

    for i = 2:N

        %%% Integrate with RK4 and 10 substeps per discrete time step
        for jj=1:10
            k1=(tau/10)*LorenzODE(state);
            k2=(tau/10)*LorenzODE(state+k1/2);
            k3=(tau/10)*LorenzODE(state+k2/2);
            k4=(tau/10)*LorenzODE(state+k3);
            state=state+k1/6+k2/3+k3/3+k4/6;
            state = state + D*sqrt(2)*sqrt(tau/10)*randn(size(state));
        end
        x(i,:,:)=state;

    end

end


function dx = LorenzODE(x)
    rho = 28; sigma = 10; beta = 8/3;
    dx = zeros(3,size(x,2));
    dx(1,:) = sigma*(x(2,:) - x(1,:));
    dx(2,:) = x(1,:).*(rho - x(3,:)) - x(2,:);
    dx(3,:) = x(1,:).*x(2,:) - beta*x(3,:);
end