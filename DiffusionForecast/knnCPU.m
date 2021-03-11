function [ds,inds] = knnCPU(R,Q,k,maxMem)
% Find the k nearest neighbors for each element of the query set Q among
% the points in the reference set R

% Q is Nxd where N is the number of query points and d is the dimension
% R is Mxd where M is the number of reference points and d is the dimension
% k is the number of nearest neighbors desired
% maxMem is the number of GB of ram knnCPU can use

% ds is Nxk and contains the distances to the k nearest neighbors.
% inds is Nxk and contains the indices of the k nearest neighbors.

    if (nargin<4)
        maxMem = 2;
    end

    M = size(R,1);
    N = size(Q,1);    
    
    maxArray = (maxMem*2500)^2;
    blockSize = floor(maxArray/M);
    blocks = floor(N/blockSize);

    ds = zeros(N,k);
    inds = zeros(N,k);

    Nr = sum(R.^2,2);
    Nq = sum(Q.^2,2);

    for b = 1:blocks
        dtemp = -2*R*Q((b-1)*blockSize + 1:b*blockSize,:)';
        dtemp = bsxfun(@plus,dtemp,Nr);
        dtemp = bsxfun(@plus,dtemp,Nq((b-1)*blockSize + 1:b*blockSize)');
        [dst,indst] = sort(dtemp,1);
        ds((b-1)*blockSize + 1:b*blockSize,:) = dst(1:k,:)';
        inds((b-1)*blockSize + 1:b*blockSize,:) = indst(1:k,:)';
    end

    if (blocks*blockSize < N)
        dtemp = -2*R*Q(blocks*blockSize + 1:N,:)';
        dtemp = bsxfun(@plus,dtemp,Nr);
        dtemp = bsxfun(@plus,dtemp,Nq(blocks*blockSize + 1:N)');
        [dst,indst] = sort(dtemp,1);
        ds(blocks*blockSize + 1:N,:) = dst(1:k,:)';
        inds(blocks*blockSize + 1:N,:) = indst(1:k,:)';
    end

    ds = real(sqrt(ds));

end



