function cnetwork = gbnn_construct_network(in_thriftymessages, out_thriftymessages, outop, inop, subsampling_p)
% cnetwork = gbnn_construct_network(in_thriftymessages [, out_thriftymessages])
% Construct/learns a network using one-shot learning. Can either construct a normal clique network, or an inter-networks map if you supply a different out_thriftymessages (different from in_thriftymessages).
%
% We simply link all characters inside each message between them as a clique, which will result in an adjacency matrix
% The network is simply an adjacency matrix of edges connections (n = l * c neurons can connect to n other neurons - in practice neurons can only connect to n - l neurons since they cannot connect to themselves + other neurons of the same cluster)
% The matrix is ordered by character position and then subordered by character value (thrifty code), eg:
% c = 3; l = 2; m = 2;
% messages = [1 2 1 ; 2 1 2];
% cnetwork =
%
%              pos1 pos2 pos3
%            __[1 2][1 2][1 2]
% pos1 1 |     1 0 0 1 1 0
%         2 |__ 0 1 1 0 0 1
% pos2 1 |     0 1 1 0 0 1
%         2 |__ 1 0 0 1 1 0__
% pos3 1 |     1 0 0 1 1 0   | Outlink character position 3 (then: first row represents value 1 at cluster 3, second row represents value 2 at cluster 3, etc.)
%         2 |__ 0 1 1 0 0 1__|
%                      [   ]
% Inlink character position 2 (then: first column represents value 1 at cluster 2, second column represents value 2 at cluster 2, etc.)
%
% pos1, pos2, pos3 = cluster1, cluster2, cluster3 (sequence trick, see the note below).
% then subordered by values (1 or 2 here because l = 2).
%
% The highlighted subsection:
% _ _ _ 1 _ _
% _ _ 1 _ _ _
% represent the messages: [_ 2 1 ; _ 1 2] (in respective order) where _ is a wildcard or undefined (because here we extracted only a part of the two messages).
%
% NOTE: The memorization of the character position in the adjacency network is because we use a sequence trick: since we use undirected edges here, to memorize sequences (eg: words), what we do is that we assign each cluster to one unique character position in the sentence: eg: cluster 1 represents character at position 1, always, cluster 2 is character at position 2, etc. Note that this trick is not used with the 2014 extension.
% NOTE2: in cluster-based networks, the adjacency matrix is symmetrical (but that's not the case with tournament-based networks).
%

    % -- Vectorized version - fastest, and so elegant!
    % We simply use a matrix product, this is greatly faster than using thriftymessages as indices
    % We also store as a logical sparse matrix to spare a lot of memory, else the matrix product will be slower than the other methods! Setting this to logical type is not necessary but it halves the memory footprint.
    % WARNING: works only with undirected clique network, but not with directed tournament-based network (Xiaoran Jiang Thesis 2013) (but it should easily work with few modifications)
    aux = gbnn_aux;

    % Default vars checking
    if ~exist('inop', 'var') || isempty(inop)
        inop = @times;
    end
    if ~exist('outop', 'var') || isempty(outop)
        outop = @sum;
    end
    if ~exist('subsampling_p', 'var') || isempty(subsampling_p) % subsampling: by default, no subsampling
        subsampling_p = 1;
        subsampling_func = @(x) x; % create a dummy identity function so that there's no subsampling
    else
        subsampling_func = @(x) subsampling(x, subsampling_p); % else we will really subsample
    end

    % Preallocating
    n = size(in_thriftymessages, 1);
    if exist('out_thriftymessages', 'var')
        m = size(out_thriftymessages, 2);
    else
        m = n;
    end
    cnetwork = logical(sparse(n,m)); % preallocating and converting to a binary sparse matrix
    if (~strcmpi(func2str(outop), 'sum') && ~strcmpi(func2str(inop), 'times')); cnetwork = double(cnetwork); end;

    % Construct the adjacency matrix (= learn the network)
    % Same network case: faster because we use only one datastructure and because MatLab/Octave recognizes the pattern a * a' and optimizes it (half the time because it knows that the computation will somehow be symmetric)
    if ~exist('out_thriftymessages', 'var')
        if ~aux.isOctave()
            in_thriftymessages = double(in_thriftymessages); % MatLab can't do matrix multiplication on logical (binary) matrices... must convert them to double beforehand
        end
        if (strcmpi(func2str(outop), 'sum') && strcmpi(func2str(inop), 'times'))
            cnetwork = logical(subsampling_func(in_thriftymessages' * in_thriftymessages)); % logical = same as min(cnetwork + thriftymessages'*thriftymessages, 1). Matrix multiplication idea by Christophe!
        else
            cnetwork = subsampling_func(gmtimes(in_thriftymessages, [], outop, inop));
        end

    % Bridge between two networks: we use a different matrix of messages for in and out (which fanals from the in network we will connect to which fanals of the out network)
    else
        if ~aux.isOctave()
            in_thriftymessages = double(in_thriftymessages); % MatLab can't do matrix multiplication on logical (binary) matrices... must convert them to double beforehand
            out_thriftymessages = double(out_thriftymessages);
        end
        if (strcmpi(func2str(outop), 'sum') && strcmpi(func2str(inop), 'times'))
            cnetwork = logical(subsampling_func(in_thriftymessages' * out_thriftymessages));
        else
            cnetwork = subsampling_func(gmtimes(in_thriftymessages', out_thriftymessages, outop, inop));
        end
    end

    % Vectorized version 2 draft: use directly the indices without matrix multiplication to avoid useless computations because of symmetry: vectorized_indices = reshape(mod(find(thriftymessages'), n), c, m)'
    % TODO: use the property of our data to speedup and spare memory: instead of matrix product, do a custom matrix logical product (where you do the product of a line and a column but then instead of the summation you do a OR, so that you end up with a boolean instead of an integer).
    % TODO: use symmetry trick by computing half of the matrix product? (row(i:c) * column(i:c) with i synchronized in both, thus that column cannot be of index lower than row).
    % TODO: reshuffle the adjacency matrix to get a band matrix? http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3148830/
    % TODO: use Locality of Reference to optimize operations?
    % TODO: use adjacency list instead of adjacency matrix?
    % TODO: use Modularity theory to efficiently use the clusters?
    % TODO: For tournament-based networks, it is also possible to use the non-symmetry by reordering the matrix: if a digraph is acyclic, then it is possible to order the points of D so that the adjacency matrix upper triangular (i.e., all positive entries are above the main diagonal).

end
