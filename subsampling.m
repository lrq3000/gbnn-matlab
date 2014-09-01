function net = subsampling(net, p)
% function net = subsampling(net, p)
% Random subsampling using a binomial process with a probability p of success per edge creation. You must input a non-binarized network (just after messages learning via matrix multiplication, and before the call to logical() so that we can know how many score each edge has)
% Used in Xiaoran thesis on tournaments to also reduce density and enhance performances.
% Note: seems to work best when data has some topology (ie: messages are not stored randomly but there's some order, eg: storing similar messages closer).
% See also: dropconnect function for a stochastic alternative.

    % Subsampling, aka deterministic DropConnect (since we will always remove 1-p links exactly)
    %idxs = shuffle(find(net));
    %idxs = idxs(1:floor(numel(idxs)*p));
    %net(idxs) = 0;

    net(net > 0) = binornd(net(net > 0), p); % the idea is that we can still use matrix multiplication to compute the adjacency matrix, and only after we subsample by replaying the scores distribution according to a binomial process (thus we can at max get the same as the original network scores, but in average it will be less)
end
