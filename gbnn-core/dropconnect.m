function net = dropconnect(net, p)
% function net = dropconnect(net, p)
% Randomly disconnect edges in a network with a probability 1-p (an edge has a probability p of persisting)
% Implementation of Wan et al's DropConnect algorithm for regularizing neural networks.
    if ~exist('p', 'var') || isempty(p); p = 0.5; end;
    %net(net > 0) = net(net > 0) .* sign(sprand(nnz(net), 1, p)); % only with Octave, because in MatLab doing net(net > 0) (in-place replacement in a sparse matrix) is VERY slow (70 seconds instead of 0.02 sec on Octave).
    vals = net(net > 0) .* sign(sprand(nnz(net), 1, p)); % compute a random mask and randomly disable some edges
    [I, J] = find(net); % find the list of activated edges before dropconnect
    net = sparse(I, J, vals, size(net,1), size(net,2)); % apply random dropconnect mask on edges and recreate a new net sparse matrix (faster than in-place replacement on MatLab)
end
