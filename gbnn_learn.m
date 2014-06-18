function [cnetwork, thriftymessages, real_density] = gbnn_learn(varargin)
%
% [cnetwork, thriftymessages, density] = gbnn_learn(m, l, c, ...,
%                                                                       Chi, cnetwork, miterator, ...
%                                                                       silent)
%
% Learns a network using one-shot learning (simply an adjacency matrix) using either a provided messages list, or either generate a random one. Returns both the network, thrifty messages and real density.
%
% This function supports named arguments, use it like this:
% gbnn_learn('m', 6, 'l', 4, 'c', 3)
%
%- m : number of messages or a matrix of messages (composed of numbers ranging from 1 to l and of length/columns c per row).
%- miterator : messages iterator, allows for out-of-core computation, meaning that you can load more messages (greater m) at the expense of more CPU (because of the loop). Set miterator <= m, and the highest allowed by your memory without running out-of-memory. Set 0 to disable.
%- l : number of character neurons (= range of values allowed per character, eg: 256 for an image in 256 shades of grey per pixel). These neurons will form one cluster (eg: 256 neurons per cluster). NOTE: only changing the number of cluster or the number of messages can change the density since density d = 1 - ( 1 - 1/l^2 )^M
%- c : cliques order = number of nodes per clique = length of messages (eg: c = 3 means that each clique will at most have 3 nodes). If Chi <= c, Chi will be set equal to c, thus c will also define the number of clusters. NOTE: c can also be a vector [min-c max-c] to enable variable length messages.
% NOTE: increasing c or decreasing miterator increase CPU usage and runtime ; increasing l or m increase memory usage.
%- Chi : number of clusters, set Chi > c to enable sparse_cliques if you want c to define the length of messages and Chi the number of clusters (where Chi must be > c) to create sparse cliques (cliques that don't use all available clusters but just c clusters per one message)
% Note: a sparse message is different from a thrifty message: a thrifty message is a message where values like 3 or 5 are replaced by thrifty codes like 00100 and 00001 (only one logical bit, all the others are sparse/zeros) and a sparse message where most clusters aren't used, like if c = 2 and Chi = 5 we will have sparse messages like 12000 or 00305. Combining a sparse message like 00305 + thrifty codes gives us: 00000 00000 00100 00000 00001 which is both thrifty and sparse (thrifty implies sparseness, but sparseness doesn't imply thriftiness. Here we have both!).
%
% == LEARNING ALGORITHM
% - Create random messages (matrix where each line is a messages composed of numbers between 1 and l, thus l is the range of values like the intensity of a pixel in an image)
% - Convert messages to sparse thrifty messages (where each number is converted to a thrifty code, eg: 3 -> [0 0 1 0] if c = 4; 2 -> [0 1 0 0 0] if c = 5)
% - Learn the network by using a simple Hebbian rule: we link together all nodes/numbers of a message, thus creating a link (here we just create the "thrifty" adjacency matrix, thrifty because we encode links relative to the thrifty messages, not the original messages, so that later we can easily push and propagate thrifty messages. So the structure of the thrifty adjacency matrix is similar to the structure of thrifty messages)
%

% == Importing some useful functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% == Arguments processing
% List of possible arguments and their default values
arguments_defaults = struct( ...
    ... % Mandatory
    'm', 0, ...
    'l', 0, ...
    'c', 0, ...
    ...
    ... % 2014 sparse enhancement
    'Chi', 0, ...
    ...
    ... % Optimization tweaks
    'cnetwork', [], ... % Reuse a previously learned network, just append new messages (lower number of messages to learn this way)
    'miterator', 0, ... % Learn messages in small batches to avoid memory overflow
    ...
    ... % Debug stuffs
    'silent', false);

% Process the arguments
arguments = aux.getnargs(varargin, arguments_defaults, true);

% Load variables into local namespace (called workspace in MatLab)
aux.varspull(arguments);

% == Sanity Checks
if m == 0 || l == 0 || c == 0
    error('Missing arguments: m, l and c are mandatory!');
end

Xlearn = [];
if ismatrix(m) && ~isscalar(m) % If user provided a matrix of messages, reuse that
    Xlearn = m; % set this into a temporary variable to hold messages
    m = size(Xlearn, 1); % m should always define the number of messages (even if as argument it can specify a matrix of messages, this is syntax sugar)
end

variable_length = false;
if isvector(c) && ~isscalar(c)
    variable_length = true;
end

% == Init data structures and other vars (need to do that before the miterator) - DO NOT TOUCH
sparse_cliques = true; % enable the creation of sparse cliques if Chi > c (cliques that don't use all available clusters but just c clusters per one message)
if Chi <= c
    Chi = c; % Chi can't be < c, thus here we ensure that
    sparse_cliques = false;
end
n = Chi * l; % total number of nodes ( = length of a message = total number of characters slots per message)
thriftymessages = logical(sparse(m,n)); % Init and converting to a binary sparse matrix
networkprovided = false;
if ~exist('cnetwork', 'var') || isempty(cnetwork) % reuse network if provided
    cnetwork = logical(sparse(n,n)); % init and converting to a binary sparse matrix
else
    networkprovided = true;
end

if miterator > m
    miterator = 0;
end

% == Show vars (just for the record)
if ~silent
    % -- Network variables
    m
    miterator
    l
    c
    Chi % -- 2014 update

    networkprovided

    % -- Custom extensions
    variable_length
end

% -- A few error checks
if numel(c) ~= 1 && numel(c) ~= 2
    error('c contains too many values! numel(c) should be equal to 1 or 2.');
end


if ~silent; totalperf = cputime(); end; % for total time perfs

% #### Learning phase
if ~silent; fprintf('#### Learning phase (construct messages and the network using Hebbian rule)\n'); aux.flushout(); end;
% == Messages iterator
% -- Init the number of loops that will be necessary to generate all the messages
mloop = 1; % 1 by default
if miterator > 0 % if the messages iterator is enabled
    mloop = ceil(m / miterator); % compute the number of loops that will be necessary to generate all the messages
end

% -- Messages Iterator start
for M = 1:mloop
    % Compute the number of messages that we will generate in this iteration
    if miterator <= 0 % 1st case: messages iterator disabled, we will generate all messages at once
        mgen = m;
    else % messages iterator enabled
        mgen = miterator; % 2nd case: generate as much as miterator allows
        if (M * miterator) > m % 3rd case: we are at the last iteration and there is less messages to generate than miterator can, we adjust to generate only the required messages
            mgen = m - ((M-1) * miterator);
        end
    end

    if ~silent; fprintf('== Generation of messages batch %i (%i messages)\n', M, mgen); aux.flushout(); end;
    % == Generate input messages
    % Generate m messages (lines) of length c or Chi (= number of columns) with a value between 1 and l
    % If Chi > c, then sparse_cliques will be enabled which will convert the messages into sparse messages (with 0 to fill the remaining length for each message).
    len = c;
    if variable_length; len = max(c); end;
    if ~isempty(Xlearn) % If user provided a matrix of messages, reuse that instead of generating a random one
        messages = Xlearn; % Use Xlearn if specified
    else % Else generate random messages
        %messages = unidrnd(l,mgen,len); % Generating messages
        messages = randi([1 l], mgen, len); % Generating messages. Use randi instead of unidrnd, the result is the same but does not necessitate the Statistics toolbox on MatLab (Octave natively supports it).
    end
    % TODO: variable_length between cmin and cmax, here we always remove 1 character from all messages
    if variable_length
        messages(messages > 0) = messages(messages > 0) - 1;
    end
    % If Chi > c, convert the messages into sparse messages by inserting 0s at random places in order to fill the length remainder (Chi-c zeros will be inserted in each message).
    if sparse_cliques
        messages = sparse([messages, sparse(mgen, Chi-c)]); % Append the 0s at the end of each message
        messages = aux.shake(messages, 2); % Then shuffle the 0s to place them randomly inside the message. This is an external FEX file, it is included in gbnn_aux.m. This just randomly shuffles the items but without shuffling the row order (so it's a per-row shuffler of columns).
    end


    % == Convert into thrifty messages
    % We convert values between 1 and l into sparse thrifty messages (constant weight code) of length l.
    % Eg: message(1,:) = [4 3 2]; sparsemessage(1,:) = [0 0 0 1 0 0 1 0 0 1 0 0]; % notice that we set 1 at the position corresponding to the value of the character at this position, and we have created submessages (thrifty codes) for each character of the message, thus if each message is of length c with each character having a range of value of l, each sparsemessage will be of length c * l)
    if ~silent;
        fprintf('-- Converting to sparse, thrifty messages\n'); aux.flushout();
        tic();
    end;

    % -- Loop version
%   for i=1:mgen % for each message
%       for j=1:c % for each character
%           thriftymessages(i+M*miterator,(j-1)*l+messages(i,j)) = 1; % expand the character into a thrifty code (set 1 where the position corresponds to the value of the character, 0 everywhere else, eg: [4] -> [0 0 0 1])
%       end
%   end

    % -- Vectorized version 1
    % The idea is to precompute two maps (the tiled messages map and indexes of the future  thriftymessages matrix) and superimpose both at once to get the final thriftymessages matrix instead of doing that in a loop.
    % The bottleneck is memory: we need to precompute two _full_ matrixes the size of the final thriftymessages matrix (there's no 0 in these two matrixes), thus you will get a memory explosion size(thriftymessages)^2 !
%   mes = kron(messages, ones(1, l)); %repmat(messages(:), 1, l)'(:)'; % expand/repeat all characters
%   idxs = repmat(1:l, m, c); % create indexes map
%   thriftymessages = (mes == idxs); % superimpose the characters repetition and the indexes map: where it matches then it's OK we have a link
    % NOTE: does not work with miterator!

    % -- Vectorized version 2
    % Same as vectorized version 1 but we spare one matrix generation by generating an index map vector (instead of a matrix) and use bsxfun to repeat it
%   mes = kron(messages, ones(1, l)); %repmat(messages(:), 1, l)'(:)'; % expand/repeat all characters
%   idxs = repmat(1:l, 1, c); % create indexes map (only a vector here instead of a matrix, we will broadcast operations using bsxfun)
%   thriftymessages = bsxfun(@eq, mes, idxs); % superimpose the characters repetition and the indexes map: where it matches then it's OK we have a link
    % NOTE: does not work with miterator!

    % -- Vectorized version 3 - Fastest! (about one-tenth of the time taken by the semi-vectorized version, plus it stays linear! and it should also be memory savvy)
    % Idea here is same as previous vectorized versions: we want to avoid generating the two maps as matrices to spare memory.
    % How we do this here is by smartly generate a vector of the indexes of each element which should be set to 1. This way we have only one vector, as long as the number of element of the messages matrix.
%    idxs_map = 0:(Chi-1); % character position index map in the thriftymessages matrix (eg: first character is in the first c numbers, second character in the c numbers after the first c numbers, etc.)
%    idxs = bsxfun(@plus, messages, l*idxs_map); % use messages matrix directly to compute the indexes (this will compute indexes independently of the row)
%    offsets = 0:(n):(mgen*n);
%    idxs = bsxfun(@plus, offsets(1:end-1)', idxs); % account for the row number now by generating a vector of index shift per row (eg: [0 lc 2lc 3lc 4lc ...]')
%    idxs = idxs + ((M-1)*miterator)*n; % offset all indexes to the current miterator position (by just skipping previous messages rows)
%    if sparse_cliques; idxs = idxs(messages > 0); end; % if sparse cliques are enabled, remove all indices of empty, zero, entries (because the indices don't care what the value is, indices of zeros entries will also be returned and scaled, but we don't wont those entries so we filter them at the end)
%    [I, J] = ind2sub([n m], idxs); % convert indexes to two subscripts vector, necessary to optimize sparse set (by using: sparsematrix = sparsematrix + sparse(...) instead of sparsematrix(idxs) = ...)
%    thriftymessages = or(thriftymessages, sparse(I, J, 1, n, m)'); % store the messages (note that the indexes we now have are columns-oriented but MatLab expects row-oriented indexes, thus we just transpose the matrix)

    % -- Vectorized version 3 moved to a function
    thriftymessages = or(thriftymessages, gbnn_messages2thrifty(messages, l, miterator, M, m));

    % -- Semi-vectorized version (consume less memory than the vectorized versions, at the expense of a bit more CPU usage linear in the number c (about the double time compared to the fastest vectorized version 3)
    % Here we vectorized only the messages loop, but we still have a complexity relative to c, but this way is quite memory savvy
%   currentmessage = max((M-1)*miterator + 1, 1);
%   for j=1:c % for each character
%       thriftymessages(sub2ind(size(thriftymessages), (currentmessage:(currentmessage-1)+mgen)', (j-1)*l + messages(:, j))) = 1; % expand all characters at this position of the message (first, then second, etc.) into a thrifty code (set 1 where the position corresponds to the value of the character, 0 everywhere else, eg: [4] -> [0 0 0 1])
%   end

    if ~silent; aux.printtime(toc()); end; % For perfs



    % == Create network = learn the network
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
    if ~silent
        fprintf('-- Learning messages into the network\n'); aux.flushout();
        tic();
    end

    % -- Loop version
%   for i=1:mgen % for each message, create a clique between all characters of this message
        % To do this, we loop through each characters to produce every combinations, and then link them together
%       for j=1:c % first character pointer
%           for k=j:c % second character pointer. TRICKS: here we compute only one part of the matrix since it is symmetric, we will just copy over the lower part from the top part.
            % NOTE: do NOT set k=j+1:c to avoid setting 1 on the diagonale (which means that a node link to itself, this is necessary when predicting so that propagation of a node stimulates the whole clique, the node itself included if it's part of the clique)
%               cnetwork((j-1)*l+messages(i,j),(k-1)*l+messages(i,k)) = 1; % link them together
                % format: cnetwork(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%           end
%       end
%   end

    % -- Semi-vectorized version
    % For each message, create a clique between all characters of this message (in the vectorized version we process all messages at once, and compute only half of the characters combinations since we can fill the remainder using the matrix symmetry)
    % To do this, we loop through each characters to produce every combinations, and then link them together
%   for j=1:c % first character pointer
%       for k=j:c % second character pointer. TRICKS: here we compute only one part of the matrix since it is symmetric, we will just copy over the lower part from the top part.
        % NOTE: do NOT set k=j+1:c to avoid setting 1 on the diagonale (which means that a node link to itself, this is necessary when predicting so that propagation of a node stimulates the whole clique, the node itself included if it's part of the clique)
%           idx = sub2ind(size(cnetwork), (j-1)*l+messages(:,j), (k-1)*l+messages(:,k)); % TRICKS: precompute the list of indexes for all messages
%           cnetwork(idx) = 1; % link the characters together
%           % format: cnetwork(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%           clear idx; % clear up some memory
%       end
%   end

    % -- Semi-vectorized version 2 - faster
    % For each message, create a clique between all characters of this message (in the vectorized version we process all messages at once, and compute only half of the characters combinations since we can fill the remainder using the matrix symmetry)
    % To do this, we loop through each characters to produce every combinations, and then link them together
%   for j=1:c % first character pointer
        % TRICKS: here we compute only one part of the matrix since it is symmetric, we will just copy over the lower part from the top part.
        % TRICKS2: we don't use the second pointer explicitly but implicitly, by precomputing all the indexes beforehand (look at the bsxfun below)
%           idx = sub2ind( size(cnetwork), repmat((j-1)*l+messages(:,j), 1, c-(j-1)), bsxfun(@plus, ((j:c)-1).*l, messages(:,j:end)) ); % TRICKS: precompute the list of indexes for all messages
%           cnetwork(idx) = 1; % link the characters together
            % format: cnetwork(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%           clear idx; % clear up some memory
%   end

    % -- Semi-vectorized version 3 - a lot faster
    % For each message, create a clique between all characters of this message (in the vectorized version we process all messages at once, and compute only half of the characters combinations since we can fill the remainder using the matrix symmetry)
    % To do this, we loop through each characters to produce every combinations, and then link them together
%   for j=1:c % first character pointer
        % TRICKS: here we compute only one part of the matrix since it is symmetric, we will just copy over the lower part from the top part.
        % TRICKS2: we don't use the second pointer explicitly but implicitly, by precomputing all the indexes beforehand (look at the bsxfun below)
        % TRICKS3: we create a sparse matrix instead of assigning 1s directly into the matrix, since it's a sparse matrix, any addition of a non-zero entry is costly. Followed the advices from http://blogs.mathworks.com/loren/2007/03/01/creating-sparse-finite-element-matrices-in-matlab/
%           cnetwork = cnetwork + sparse(repmat((j-1)*l+messages(:,j), 1, c-(j-1)), bsxfun(@plus, ((j:c)-1).*l, messages(:,j:end)), 1, n, n);
            % format: cnetwork(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%   end

    % -- Semi-vectorized version 4 - fastest after the vectorized version, and is a bit more memory consuming
    % For each message, create a clique between all characters of this message (in the vectorized version we process all messages at once, and compute only half of the characters combinations since we can fill the remainder using the matrix symmetry)
    % To do this, we loop through each characters to produce every combinations, and then link them together
%   combinations_count = sum(1:c);
%   rows = zeros(mgen, combinations_count);
%   cols = zeros(mgen, combinations_count);
%   parfor j=1:c % first character pointer
        % TRICKS: here we compute only one part of the matrix since it is symmetric, we will just copy over the lower part from the top part.
        % TRICKS2: we don't use the second pointer explicitly but implicitly, by precomputing all the indexes beforehand (look at the bsxfun below)
        % TRICKS3: we create a sparse matrix instead of assigning 1s directly into the matrix, since it's a sparse matrix, any addition of a non-zero entry is costly. Followed the advices from http://blogs.mathworks.com/loren/2007/03/01/creating-sparse-finite-element-matrices-in-matlab/
        % TRICKS4: precache all indexes and then add the new links all at once after the loop. A bit more memory consuming, but so much faster!
%           idxend = sum(c:-1:(c-(j-1))); % combinatorial indexing (decreases with higher j)
%           idxstart = idxend - (c-j);
%           rows(:, idxstart:idxend) = repmat((j-1)*l+messages(:,j), 1, c-(j-1));
%           cols(:, idxstart:idxend) = bsxfun(@plus, ((j:c)-1).*l, messages(:,j:end));
            % format: cnetwork(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%   end
    % At the end, add all new links at once
%   cnetwork = cnetwork + sparse(rows, cols, 1, n, n);

    % -- Vectorized version - fastest, and so elegant!
    % We simply use a matrix product, this is greatly faster than using thriftymessages as indices
    % We also store as a logical sparse matrix to spare a lot of memory, else the matrix product will be slower than the other methods! Setting this to logical type is not necessary but it halves the memory footprint.
    % WARNING: works only with undirected clique network, but not with directed tournament-based network (Xiaoran Jiang Thesis 2013)
    if M == 1 && ~networkprovided % case when network is empty, this is faster
        if aux.isOctave()
            cnetwork = logical(thriftymessages' * thriftymessages); % Credits go to Christophe for the tip!
        else % MatLab can't do matrix multiplication on logical (binary) matrices... must convert them to double beforehand
            dthriftymessages = double(thriftymessages);
            cnetwork = logical(dthriftymessages' * dthriftymessages);
        end
    else % case when we iteratively append new messages (either because of miterator or because user provided a network to reuse), we update the previous network
        if aux.isOctave()
            cnetwork = or(cnetwork, logical(thriftymessages' * thriftymessages)); % same as min(cnetwork + thriftymessages'*thriftymessages, 1)
        else
            dthriftymessages = double(thriftymessages);
            cnetwork = or(cnetwork, logical(dthriftymessages' * dthriftymessages)); % same as min(cnetwork + thriftymessages'*thriftymessages, 1)
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

    if ~silent; aux.printtime(toc()); end; % just to show performance, learning the network (adjacency matrix) _was_ the bottleneck

    if miterator > 0 % for debug cases, it may be nice to keep messages to compare with thriftymessages and network
        clear messages; % clear up some memory
    end
end
%cnetwork = cnetwork + cnetwork'; % TRICKS: using matrix symmetry to fill the remainder. NOTE: when using miterator, you need to fill the symmetry only at the end of the loop, else you will run out of memory
%cnetwork = logical(cnetwork); % = min(cnetwork,1) threshold because values can only be 1 at max (but if there were duplicate messages, some entries can have a higher value than 1)
%cnetwork = or(cnetwork, cnetwork'); % not necessary when using the vectorized version
% Note that  or(cnetwork, cnetwork') = min(cnetwork + cnetwork', 1) but the latter is a bit slower on big datasets
thriftymessages = logical(thriftymessages); % NOTE: prefer logical(x) rather than min(x, 1) because: logical is faster, and also the ending data will take less storage space (about half)

% Clean up memory, else MatLab/Octave keep everything in memory
clear messages;

if ~silent; fprintf('-- Finished learning!\n'); aux.flushout(); end;

% count_cliques = sum(cnetwork^2 == c) % count the total number of cliques. WARNING: this is VERY slow but there's no better way to my knowledge



% == Compute density and some practical and theoretical stats
real_density = full(  (sum(sum(cnetwork)) - sum(diag(cnetwork))) / (Chi*(Chi-1) * l^2)  );
if ~silent
    fprintf('-- Computing density\n'); aux.flushout();
    real_density % density = (number_of_links - loops) / max_number_of_links; where max_number_of_links = (Chi*(Chi-1) * l^2).
    theoretical_average_density = 1 - (1 - c * (c-1) / (Chi * (Chi - 1) * l^2) )^m
    total_number_of_messages_really_stored = log2(1 - real_density) / log2(1 - c*(c-1) / (Chi*(Chi-1)*l.^2))
    number_of_nodes = n % total number of nodes (l * c)
    number_of_active_links = (nnz(cnetwork) - nnz(diag(cnetwork))) / 2 % total number of active links
    number_of_possible_links = Chi*(Chi-1) * l.^2 / 2 % divide by 2 here because edges are undirected. In a tournament-based network you can remove the /2. This is also the capacity of the network (ie: the number of information bits the network can store)
    B_theoretical_information = m * (log2(nchoosek(Chi, c)) + c * log2(l))
    %real_info_stored = m * 2^c;
    theoretical_efficiency = B_theoretical_information / number_of_possible_links % efficiency = B / Q = 2*m * (log2(nchoosek(Chi, c)) + c * log2(l)) / (Chi*(Chi-1) * l.^2)
    %real_efficiency = real_info_stored / number_of_active_links % TODO: how to get current real information?
    Mmax = number_of_possible_links / (2 * (log2(nchoosek(Chi, c)) + c * log2(l)))
end

if ~silent; aux.printcputime(cputime - totalperf, 'Total elapsed cpu time for learning is %g seconds.\n'); aux.flushout(); end;

if ~silent; fprintf('=> Learning done!\n'); aux.flushout(); end;

end % endfunction
