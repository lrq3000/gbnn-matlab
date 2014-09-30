function thriftymessages = gbnn_messages2thrifty(messages, l, miterator, M, m)
% thriftymessages = gbnn_messages2thrifty(messages, l)
% Converts a messages matrix into a sparse thrifty messages matrix.
% Messages matrix should be of size [m, Chi] (m lines of messages of length Chi each).
% % We convert values between 1 and l into sparse thrifty messages (constant weight code) of length l.
% Eg of the result: message(1,:) = [4 3 2]; sparsemessage(1,:) = [0 0 0 1 0 0 1 0 0 1 0 0]; % notice that we set 1 at the position corresponding to the value of the character at this position, and we have created submessages (thrifty codes) for each character of the message, thus if each message is of length c with each character having a range of value of l, each sparsemessage will be of length c * l)
%

    if ~exist('miterator', 'var')
        miterator = 0;
    end
    if ~exist('M', 'var')
        M = 0;
    end

    [mgen, Chi] = size(messages);
    n = Chi*l;
    
    if ~exist('m', 'var')
        m = mgen;
    end

    % -- Vectorized version 3 - Fastest! (about one-tenth of the time taken by the semi-vectorized version, plus it stays linear! and it should also be memory savvy)
    % The original idea was to precompute two maps (the tiled messages map and indexes of the future thriftymessages matrix) and superimpose both at once to get the final thriftymessages matrix instead of doing that in a loop.
    % Idea here is similar, except that we want to avoid generating the two maps as matrices to spare memory.
    % How we do this is by smartly generate a vector of the indexes of each element which should be set to 1. This way we have only one vector, as long as the number of element of the messages matrix.
    idxs_map = 0:(Chi-1); % character position index map in the sparsemessages matrix (eg: first character is in the first c numbers, second character in the c numbers after the first c numbers, etc.)
    idxs = bsxfun(@plus, messages, l*idxs_map); % use messages matrix directly to compute the indexes (this will compute indexes independently of the row)
    offsets = 0:(n):(mgen*n);
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