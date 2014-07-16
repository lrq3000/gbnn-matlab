function net = dropconnect(net, p)
% function net = dropconnect(net, p)
% Randomly disconnect edges in a network with a probability 1-p (an edge has a probability p of persisting)
% Implementation of Wan et al's DropConnect algorithm for regularizing neural networks.
    if ~exist('p', 'var') || isempty(p); p = 0.5; end;
    net(net > 0) = net(net > 0) .* sign(sprand(nnz(net), 1, p));
end
