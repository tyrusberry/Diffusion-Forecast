function [x,y,z,t,theta,phi] = torusSDE(theta0,phi0,T,tau,D)

t = 0:tau:T-tau;
N = length(t);

%%% intrinsic coordinates in R^2 (periodic)
theta = zeros(N,size(theta0,2));
phi = zeros(N,size(phi0,2));

theta(1,:) = theta0;
phi(1,:) = phi0;

for i = 2:N
        substeps = 10;
        tx = theta(i-1,:);
        ty = phi(i-1,:);
        for j = 1:substeps
            
            tx = tx + (2+cos(tx).*cos(2*ty)/2+cos(tx+pi/2))*(tau/substeps)/4;
            ty = ty + (10+cos(tx+ty/2)/2+cos(tx+pi/2))*(tau/substeps);
            
            stoch = D*sqrt((tau/substeps))*randn(2,size(theta0,2));

            tx = tx + (1+sin(tx)).*stoch(1,:)/4 + cos(tx+ty).*stoch(2,:)/4;
            ty = ty + cos(tx+ty).*stoch(1,:)/4 + (1+sin(ty).*cos(tx)).*stoch(2,:)/40;
            
        end
        theta(i,:) = mod(tx,2*pi);
        phi(i,:) = mod(ty,2*pi);
end

%%% Curved Torus in R^3
R=2; r=1;
x=(R+r*sin(theta)).*cos(phi);
y=(R+r*sin(theta)).*sin(phi);
z=r.*cos(theta);