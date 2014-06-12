function thriftymessages = gbnn_messages2thrifty(messages, l, miterator, M)
% thriftymessages = gbnn_messages2thrifty(messages, l)
% Converts a messages matrix into a sparse thrifty messages matrix.
% Messages matrix should be of size [m, Chi] (m lines of messages of length Chi each).

    if ~exist('miterator', 'var')
        miterator = 0;
    end
    if ~exist('M', 'var')
        M = 0;
    end

    [m, Chi] = size(messages);
    n = Chi*l;

    idxs_map = 0:(Chi-1); % character position index map in the sparsemessages matrix (eg: first character is in the first c numbers, second character in the c numbers after the first c numbers, etc.)
    idxs = bsxfun(@plus, messages, l*idxs_map); % use messages matrix directly to compute the indexes (this will compute indexes independently of the row)
    offsets = 0:(n):(m*n);
    idxs = bsxfun(@plus, offsets(1:end-1)', idxs); % account for the row number now by generating a vector of index shift per row (eg: [0 lc 2lc 3lc 4lc ...]')
    idxs = idxs + ((M-1)*miterator)*n; % offset all indexes to the current miterator position (by just skipping previous messages rows)
    idxs = idxs(messages > 0); % if sparse cliques are enabled, remove all indices of empty, zero, entries (because the indices don't care what the value is, indices of zeros entries will also be returned and scaled, but we don't wont those entries so we filter them at the end)
    [I, J] = ind2sub([n m], idxs); % convert indexes to two subscripts vector, necessary to optimize sparse set (by using: sparsematrix = sparsematrix + sparse(...) instead of sparsematrix(idxs) = ...)
    thriftymessages =  sparse(I, J, 1, n, m)'; % store the messages (note that the indexes we now have are columns-oriented but MatLab expects row-oriented indexes, thus we just transpose the matrix)

    function ok_flag = compare_messages_thrifty(messages, thriftymessages, l, Chi)
    % Just a quick unit-testing function to check that the thriftymessages matrix is equivalent to the original messages matrix
        tt = reshape(thriftymessages', l, []);
        tt(tt > 0) = find(tt);
        tt(tt > 0) = mod(tt(tt > 0)-1, l)+1;
        tt = reshape(max(tt), Chi, [])';
        ok_flag = (nnz(tt == messages) == numel(messages));
    end

end