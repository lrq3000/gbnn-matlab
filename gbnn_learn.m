function [network, sparsemessages, real_density] = gbnn_learn(network, m, miterator, l, c, Chi, ...
                                                                                                        variable_length, ...
                                                                                                        silent, debug)
%
% [network, sparsemessages, density] = gbnn_learn(network, m, miterator, l, c, Chi, ...
%                                                                                                        variable_length, ...
%                                                                                                        silent, debug)
%
%
%
% #### Gripon-Berrou Neural Network, aka thrifty clique codes. Implementation in MatLab, by Vincent Gripon. Coded and tested on Octave 3.6.4 Windows and MatLab 2013a Linux.
% Optimized (compatibility Octave/MatLab, vectorizations, binary sparse matrices, etc.), comments additions and functionality extensions by Stephen Larroque
% To run the code even faster, first use MatLab (which has a sparse-aware bsxfun() function) and use libraries replacements advised in https://research.microsoft.com/en-us/um/people/minka/software/matlab.html and also sparse2 instead of sparse as advised at http://blogs.mathworks.com/loren/2007/03/01/creating-sparse-finite-element-matrices-in-matlab/
% The code is also parallelizable, just enable multithreading in your MatLab config. You can also enable the use of GPU for implicit GPU parallelization in MatLab and/or for explicit GPU parallelization (to enable the for loops or if you want to convert some of the code to explicitly process on GPU) you can use third-party libraries like GPUmat for free or ArrayFire for maximum speedup.
%
% -- Data variables
%- Xlearn : messages matrix to learn from. Set [] to use a randomly generated set of messages. Format: m*c (1 message per line and message length of c), and values should be in the range 1:l. This matrix will automatically be converted to a sparse messages matrix.
%- Xtest : sparse messages matrix to decode from (the messages should be complete without erasures, these will be made by the algorithm, so that we can evaluate performances test!). Set [] to use Xlearn. Format: attention this is a sparse matrix, it's not the same format as Xlearn, you have to convert every character into a thrifty code.
%
% -- Network variables
%- m : number of messages
%- miterator : messages iterator, allows for out-of-core computation, meaning that you can load more messages (greater m) at the expense of more CPU (because of the loop). Set miterator <= m, and the highest allowed by your memory without running out-of-memory. Set 0 to disable.
%- l : number of character neurons (= range of values allowed per character, eg: 256 for an image in 256 shades of grey per pixel). These neurons will form one cluster (eg: 256 neurons per cluster). NOTE: only changing the number of cluster or the number of messages can change the density since density d = 1 - ( 1 - 1/l^2 )^M
%- c : cliques order = number of nodes per clique = length of messages (eg: c = 3 means that each clique will at most have 3 nodes). If Chi <= c, Chi will be set equal to c, thus c will also define the number of clusters.
% NOTE: increasing c or decreasing miterator increase CPU usage and runtime ; increasing l or m increase memory usage.
%
% -- Test variables
%- erasures : number of characters that will be erased (or noised if tampering_type == "noise") from a message for test. NOTE: must be lower than c!
%- iterations : number of iterations to let the network converge
%- tampered_messages_per_test : number of tampered messages to generate and try per test (for maximum speed, set this to the maximum allowed by your memory and decrease the number tests)
%- tests : number of tests (number of batch of tampered messages to test)
% NOTE2: at test phase, iteration 1 will always be very very fast, and subsequent iterations (but mostly the second one) will be very slow in comparison. This is normal because after the first propagation, the number of activated nodes will be hugely bigger (because one node is connected to many others), thus producing this slowing effect.
% NOTE3: setting a high erasures number will slow down the test phase after first iteration, because it will exacerbate the number of potential candidates (ie: the maximum score will be lower since there is less activated nodes at the first propagation iteration, and thus many candidate nodes will attain the max score, and thus will be selected by WTA). This is an extension of the effect described in note2.
%
% -- 2014 update
%- Chi : number of clusters, set Chi > c to enable sparse_cliques if you want c to define the length of messages and Chi the number of clusters (where Chi must be > c) to create sparse cliques (cliques that don't use all available clusters but just c clusters per one message)
%- guiding_mask : guide the decoding by focusing only on the clusters where we know the characters may be (by cheating and using the initial message as a guide, but a less efficient guiding mask could be generated by other means). This is only useful if you enable sparse_cliques (Chi > c).
%- gamma_memory : memory effect: keep a bit of the previous nodes value. This helps when the original message is sure to contain the correct bits (ie: works with messages partially erased, but maybe not with noise)
%- threshold : activation threshold. Nodes having a score below this threshold will be deactivated (ie: the nodes won't produce a spike). Unlike the article, this here is just a scalar (a global threshold for all nodes), rather than a vector containing per-cluster thresholds.
%- propagation_rule : also called "dynamic rule", defines the type of propagation algorithm to use (how nodes scores will be computed at next iteration). "sum"=Sum-of-Sum ; "normalized"=Normalized-Sum-of-Sum ; "max"=Sum-of-Max % TODO: only 0 SoS is implemented right now!
%- filtering_rule : also called "activation rule", defines the type of filtering algorithm (how to select the nodes that should remain active), generally a Winner-take-all algorithm in one of the following (case insensitive): 'WTA'=Winner-take-all (keep per-cluster nodes with max activation score) ; 'kWTA'=k-Winners-take-all (keep exactly k best nodes per cluster) ; 'oGWTA'=One-Global-Winner-take-all (same as GWTA but only one node per message can stay activated, not all nodes having the max score) ; 'GWTA'=Global Winner-take-all (keep nodes that have the max score in the whole network) ; 'GkWTA'=Global k-Winner-take-all (keep nodes that have a score equal to one of k best scores) ; 'WsTA'=WinnerS-take-all (per cluster, select the kth best score, and accept all nodes with a score greater or equal to this score - similar to kWTA but all nodes with the kth score are accepted, not only a fixed number k of nodes) ; 'GWsTA'=Global WinnerS-take-all (same as WsTA but per the whole message) ; 'GLKO'=Global Loser-kicked-out (kick nodes that have the worst score globally) ; 'GkLKO'=Global k-Losers-kicked-out (kick all nodes that have a score equal to one of the k worst scores globally) ; 'LKO'=Losers-Kicked-Out (locally, kick k nodes with worst score per-cluster) ; 'kLKO'=k-Losers-kicked-out (kick k nodes with worst score per-cluster) ; 'CGkWTA'=Concurrent Global k-Winners-Take-All (kGWTA + trimming out all scores that are below the k-th max score) ; TODO : 10=Optimal-Global-Loser-Kicked-Out (GkLKO but with k = 1 to kick only one node per iteration)
%- tampering_type : type of message tampering in the tests. "erase"=erasures (random activated bits are switched off) ; "noise"=noise (random bits are flipped, so a 0 becomes 1 and a 1 becomes 0).
%
% -- Custom extensions
%- residual_memory : residual memory: previously activated nodes lingers a bit and participate in the next iteration
%- variable_length : variable length messages (with a varying c number of characters)
%- concurrent_cliques : allow to decode multiple messages concurrently (can specify the number here)
%- debug : clean up some more memory if disabled
%
%
% #### Thrifty clique codes algorithm ####
% (quite simple, it's a kind of associative memory)
%
% == LEARNING
% - Create random messages (matrix where each line is a messages composed of numbers between 1 and l, thus l is the range of values like the intensity of a pixel in an image)
% - Convert messages to sparse thrifty messages (where each number is converted to a thrifty code, eg: 3 -> [0 0 1 0] if c = 4; 2 -> [0 1 0 0 0] if c = 5)
% - Learn the network by using a simple Hebbian rule: we link together all nodes/numbers of a message, thus creating a link (here we just create the "thrifty" adjacency matrix, thrifty because we encode links relative to the thrifty messages, not the original messages, so that later we can easily push and propagate thrifty messages. So the structure of the thrifty adjacency matrix is similar to the structure of thrifty messages)
%
% == PREDICTION
% - Generate tampered messages with random erasures of randomly selected characters
% - Try to recover the original messages using the network. In a loop, do:
%    * Update network state (global decoding): Push and propagate tampered messages into the network (a simple matrix product like in Hopfield's network)
%    * Keep winners, most likely characters (local thrifty code): Use winner-takes-all on each cluster: only winners (having max score) must remain active on each cluster. This is equivalent to say that for each character position in the message, we keep the character that has max score (eg: [2 1 0 1] with c = 2 and l = 2, the winners in the thrifty message will be [1 0 0 1] which can be converted to the full message [1 2])
%    * Compute error score: if there is at least one wrong character, error = error + 1
%
%
% TROUBLESHOOTING:
%
% - At Learning messages into the network, you encounter the following error: memory exhausted
%   This means that your software hasn't enough contiguous memory to run the program.
%    If you use Octave (as of 2014, v3.6.4), you cannot do anything about it. The first reason is that Octave does not possess a sparse bsxfun yet, thus everytime bsxfun is used, all input datatypes are converted to full, and it also returns a full datatype, which is why you run out of memory. Also, there is a hard limit for memory which is 2GB, and there's nothing you can do about it. In this case, the only thing you can do is to lower the m, l and c variables.
%    If you use MatLab, you can increase your virtual memory file (swap), MatLab is not limited and can use any contiguous memory available. Also, its bsxfun implementation is totally sparse compatible, thus the code should be a lot faster and memory efficient than with Octave (but this may change in the future).
%
% - At Converting to sparse, thrifty messages, you encounter the error: ind2sub: subscript indices must be either positive integers or logicals
%   This means that the matrix you try to create is simply too big, the indices overflow the limit (even if there's no content!). If you use Octave (as of 2014), there's nothing you can do, matrices are limited by the number of indices they can have. If you use MatLab, there should be no such limitations.
%
% - I have some troubles understanding the code
%   The code has been commented as much as possible, but since it's as much vectorized as was possible by the authors, it's understandably a bit cryptic in some parts. If you have some troubles understanding what some parts of the code do, you can try to place the "keyboard" command just before the line that upsets you, then run the code. This will stop the code at runtime right where you placed the keyboard command, and let you print the various variables so that you can get a clearer idea of what's going on. Also feel free to decompose some complex inlined commands, by typing them yourself step by step to monitor what's going on exactly and get a better grasp of the data structures manipulations involved.
%   Also you should try with a small network and test set (low values between 2 and 6 for m, l, c and tampered_messages_per_test), and try to understand first how the sparsemessages and network are built since they are the core of this algorithm.
%   Also you can place a keyboard command at the end of the script, call one run and then use full(reshape([init ; out ; propag], n, [])) to show the evolution of the decoding process.
%   Finally, if you use keyboard, don't forget to "clear all" everytime before doing a run, else MatLab/Octave may cache the function and may not see your latest changes.
%
% - When I use Global (k-)Loser-Kicked-Out, my error rate is very high!
%   First check your gamma_memory value: if it's enabled, try to set it to 0. Indeed, what this does is that the network always remember the first input (partially erased message), so that the already known characters are always largely winning over recovered characters, and are thus kicked out by this type of filtering rule. If you disable the memory, this should avoid this kind of effect.
%
% TODO:
% - Implement sparse_cliques (Chi) in gbnn_mini
% - variable_length complete implementation and tests
% - propagation_rule 1 and 2 (normalized and Sum of Max)
% - convergence stop criterions: NoChange (no change in out messages between two iterations, just like Perceptron) and Clique (all nodes have the same value, won't work with concurrent_cliques)
%

% == Importing some useful functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% == Init variables
Xlearn = [];
if ismatrix(m) && ~isscalar(m) % If user provided a matrix of messages, reuse that
    Xlearn = m; % set this into a temporary variable to hold messages
    m = size(Xlearn, 1); % m should always define the number of messages (even if as argument it can specify a matrix of messages, this is syntax sugar)
end

if ~exist('Chi', 'var') || isempty(Chi)
    Chi = c;
end

if ~exist('variable_length', 'var') || isempty(variable_length)
    variable_length = false;
end

if ~exist('silent', 'var')
    silent = false;
end
if ~exist('debug', 'var')
    debug = false;
end


% == Show vars (just for the record)
if ~silent
    % -- Network variables
    m
    miterator
    l
    c
    Chi % -- 2014 update

    % -- Custom extensions
    variable_length
    debug
end



% == Init data structures and other vars (need to do that before the miterator) - DO NOT TOUCH
sparse_cliques = true; % enable the creation of sparse cliques if Chi > c (cliques that don't use all available clusters but just c clusters per one message)
if Chi <= c
    Chi = c; % Chi can't be < c, thus here we ensure that
    sparse_cliques = false;
end
n = Chi * l; % total number of nodes ( = length of a message = total number of characters slots per message)
sparsemessages = logical(sparse(m,n)); % Init and converting to a binary sparse matrix
networkprovided = false;
if ~exist('network', 'var') || isempty(network) % reuse network if provided
    network = logical(sparse(n,n)); % init and converting to a binary sparse matrix
else
    networkprovided = true;
end

if miterator > m
    miterator = 0;
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
    % Generate m messages (lines) of length c (columns) with a value between 1 and l
    len = c;
    if variable_length; len = Chi; end;
    if ~isempty(Xlearn) % If user provided a matrix of messages, reuse that instead of generating a random one
        messages = Xlearn; % Use Xlearn if specified
    else % Else generate random messages
        %messages = unidrnd(l,mgen,len); % Generating messages
        messages = randi([1 l], mgen, len); % Generating messages. Use randi instead of unidrnd, the result is the same but does not necessitate the Statistics toolbox on MatLab (Octave natively supports it).
    end
    if variable_length
        messages = sparse(messages(messages > 0) - 1);
    end
    if sparse_cliques
        messages = [messages, sparse(mgen, Chi-c)];
        messages = aux.shake(messages, 2); % this is an external FEX file, please download it if you don't have it! This just randomly shuffles the items but without shuffling the row order (so it's a per-row shuffler of columns).
    end


    % == Convert into sparse messages
    % We convert values between 1 and l into sparse thrifty messages (constant weight code) of length l.
    % Eg: message(1,:) = [4 3 2]; sparsemessage(1,:) = [0 0 0 1 0 0 1 0 0 1 0 0]; % notice that we set 1 at the position corresponding to the value of the character at this position, and we have created submessages (thrifty codes) for each character of the message, thus if each message is of length c with each character having a range of value of l, each sparsemessage will be of length c * l)
    if ~silent;
        fprintf('-- Converting to sparse, thrifty messages\n'); aux.flushout();
        tic();
    end;

    % -- Loop version
%   for i=1:mgen % for each message
%       for j=1:c % for each character
%           sparsemessages(i+M*miterator,(j-1)*l+messages(i,j)) = 1; % expand the character into a thrifty code (set 1 where the position corresponds to the value of the character, 0 everywhere else, eg: [4] -> [0 0 0 1])
%       end
%   end

    % -- Vectorized version 1
    % The idea is to precompute two maps (the tiled messages map and indexes of the future  sparsemessages matrix) and superimpose both at once to get the final sparsemessages matrix instead of doing that in a loop.
    % The bottleneck is memory: we need to precompute two _full_ matrixes the size of the final sparsemessages matrix (there's no 0 in these two matrixes), thus you will get a memory explosion size(sparsemessages)^2 !
%   mes = kron(messages, ones(1, l)); %repmat(messages(:), 1, l)'(:)'; % expand/repeat all characters
%   idxs = repmat(1:l, m, c); % create indexes map
%   sparsemessages = (mes == idxs); % superimpose the characters repetition and the indexes map: where it matches then it's OK we have a link
    % NOTE: does not work with miterator!

    % -- Vectorized version 2
    % Same as vectorized version 1 but we spare one matrix generation by generating an index map vector (instead of a matrix) and use bsxfun to repeat it
%   mes = kron(messages, ones(1, l)); %repmat(messages(:), 1, l)'(:)'; % expand/repeat all characters
%   idxs = repmat(1:l, 1, c); % create indexes map (only a vector here instead of a matrix, we will broadcast operations using bsxfun)
%   sparsemessages = bsxfun(@eq, mes, idxs); % superimpose the characters repetition and the indexes map: where it matches then it's OK we have a link
    % NOTE: does not work with miterator!

    % -- Vectorized version 3 - Fastest! (about one-tenth of the time taken by the semi-vectorized version, plus it stays linear! and it should also be memory savvy)
    % Idea here is same as previous vectorized versions: we want to avoid generating the two maps as matrices to spare memory.
    % How we do this here is by smartly generate a vector of the indexes of each element which should be set to 1. This way we have only one vector, as long as the number of element of the messages matrix.
    idxs_map = 0:(Chi-1); % character position index map in the sparsemessages matrix (eg: first character is in the first c numbers, second character in the c numbers after the first c numbers, etc.)
    idxs = bsxfun(@plus, messages, l*idxs_map); % use messages matrix directly to compute the indexes (this will compute indexes independently of the row)
    offsets = 0:(n):(mgen*n);
    idxs = bsxfun(@plus, offsets(1:end-1)', idxs); % account for the row number now by generating a vector of index shift per row (eg: [0 lc 2lc 3lc 4lc ...]')
    idxs = idxs + ((M-1)*miterator)*n; % offset all indexes to the current miterator position (by just skipping previous messages rows)
    if sparse_cliques; idxs = idxs(messages > 0); end; % if sparse cliques are enabled, remove all indices of empty, zero, entries (because the indices don't care what the value is, indices of zeros entries will also be returned and scaled, but we don't wont those entries so we filter them at the end)
    [I, J] = ind2sub([n m], idxs); % convert indexes to two subscripts vector, necessary to optimize sparse set (by using: sparsematrix = sparsematrix + sparse(...) instead of sparsematrix(idxs) = ...)
    sparsemessages = or(sparsemessages, sparse(I, J, 1, n, m)'); % store the messages (note that the indexes we now have are columns-oriented but MatLab expects row-oriented indexes, thus we just transpose the matrix)

    % -- Semi-vectorized version (consume less memory than the vectorized versions, at the expense of a bit more CPU usage linear in the number c (about the double time compared to the fastest vectorized version 3)
    % Here we vectorized only the messages loop, but we still have a complexity relative to c, but this way is quite memory savvy
%   currentmessage = max((M-1)*miterator + 1, 1);
%   for j=1:c % for each character
%       sparsemessages(sub2ind(size(sparsemessages), (currentmessage:(currentmessage-1)+mgen)', (j-1)*l + messages(:, j))) = 1; % expand all characters at this position of the message (first, then second, etc.) into a thrifty code (set 1 where the position corresponds to the value of the character, 0 everywhere else, eg: [4] -> [0 0 0 1])
%   end

    if ~silent; aux.printtime(toc()); end; % For perfs



    % == Create network = learn the network
    % We simply link all characters inside each message between them as a clique, which will result in an adjacency matrix
    % The network is simply an adjacency matrix of edges connections (n = l * c neurons can connect to n other neurons - in practice neurons can only connect to n - l neurons since they cannot connect to themselves + other neurons of the same cluster)
    % The matrix is ordered by character position and then subordered by character value (thrifty code), eg:
    % c = 3; l = 2; m = 2;
    % messages = [1 2 1 ; 2 1 2];
    % network =
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
%               network((j-1)*l+messages(i,j),(k-1)*l+messages(i,k)) = 1; % link them together
                % format: network(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%           end
%       end
%   end

    % -- Semi-vectorized version
    % For each message, create a clique between all characters of this message (in the vectorized version we process all messages at once, and compute only half of the characters combinations since we can fill the remainder using the matrix symmetry)
    % To do this, we loop through each characters to produce every combinations, and then link them together
%   for j=1:c % first character pointer
%       for k=j:c % second character pointer. TRICKS: here we compute only one part of the matrix since it is symmetric, we will just copy over the lower part from the top part.
        % NOTE: do NOT set k=j+1:c to avoid setting 1 on the diagonale (which means that a node link to itself, this is necessary when predicting so that propagation of a node stimulates the whole clique, the node itself included if it's part of the clique)
%           idx = sub2ind(size(network), (j-1)*l+messages(:,j), (k-1)*l+messages(:,k)); % TRICKS: precompute the list of indexes for all messages
%           network(idx) = 1; % link the characters together
%           % format: network(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%           clear idx; % clear up some memory
%       end
%   end

    % -- Semi-vectorized version 2 - faster
    % For each message, create a clique between all characters of this message (in the vectorized version we process all messages at once, and compute only half of the characters combinations since we can fill the remainder using the matrix symmetry)
    % To do this, we loop through each characters to produce every combinations, and then link them together
%   for j=1:c % first character pointer
        % TRICKS: here we compute only one part of the matrix since it is symmetric, we will just copy over the lower part from the top part.
        % TRICKS2: we don't use the second pointer explicitly but implicitly, by precomputing all the indexes beforehand (look at the bsxfun below)
%           idx = sub2ind( size(network), repmat((j-1)*l+messages(:,j), 1, c-(j-1)), bsxfun(@plus, ((j:c)-1).*l, messages(:,j:end)) ); % TRICKS: precompute the list of indexes for all messages
%           network(idx) = 1; % link the characters together
            % format: network(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%           clear idx; % clear up some memory
%   end

    % -- Semi-vectorized version 3 - a lot faster
    % For each message, create a clique between all characters of this message (in the vectorized version we process all messages at once, and compute only half of the characters combinations since we can fill the remainder using the matrix symmetry)
    % To do this, we loop through each characters to produce every combinations, and then link them together
%   for j=1:c % first character pointer
        % TRICKS: here we compute only one part of the matrix since it is symmetric, we will just copy over the lower part from the top part.
        % TRICKS2: we don't use the second pointer explicitly but implicitly, by precomputing all the indexes beforehand (look at the bsxfun below)
        % TRICKS3: we create a sparse matrix instead of assigning 1s directly into the matrix, since it's a sparse matrix, any addition of a non-zero entry is costly. Followed the advices from http://blogs.mathworks.com/loren/2007/03/01/creating-sparse-finite-element-matrices-in-matlab/
%           network = network + sparse(repmat((j-1)*l+messages(:,j), 1, c-(j-1)), bsxfun(@plus, ((j:c)-1).*l, messages(:,j:end)), 1, n, n);
            % format: network(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
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
            % format: network(position-char1 * l (range of values) + char1, position-char2 * l + char2); where position-charx is the position of the character in the message (first character, second character, etc), and charx is the value of the character (between 1 and l)
%   end
    % At the end, add all new links at once
%   network = network + sparse(rows, cols, 1, n, n);

    % -- Vectorized version - fastest, and so elegant!
    % We simply use a matrix product, this is greatly faster than using sparsemessages as indices
    % We also store as a logical sparse matrix to spare a lot of memory, else the matrix product will be slower than the other methods! Setting this to logical type is not necessary but it halves the memory footprint.
    % WARNING: works only with undirected clique network, but not with directed tournament-based network (Xiaoran Jiang Thesis 2013)
    if M == 1 && ~networkprovided % case when network is empty, this is faster
        if aux.isOctave()
            network = logical(sparsemessages' * sparsemessages); % Credits go to Christophe for the tip!
        else % MatLab can't do matrix multiplication on logical (binary) matrices... must convert them to double beforehand
            dsparsemessages = double(sparsemessages);
            network = logical(dsparsemessages' * dsparsemessages);
        end
    else % case when we iteratively append new messages (either because of miterator or because user provided a network to reuse), we update the previous network
        if aux.isOctave()
            network = or(network, logical(sparsemessages' * sparsemessages)); % same as min(network + sparsemessages'*sparsemessages, 1)
        else
            dsparsemessages = double(sparsemessages);
            network = or(network, logical(dsparsemessages' * dsparsemessages)); % same as min(network + sparsemessages'*sparsemessages, 1)
        end
    end
    % Vectorized version 2 draft: use directly the indices without matrix multiplication to avoid useless computations because of symmetry: vectorized_indices = reshape(mod(find(sparsemessages'), n), c, m)'
    % TODO: use the property of our data to speedup and spare memory: instead of matrix product, do a custom matrix logical product (where you do the product of a line and a column but then instead of the summation you do a OR, so that you end up with a boolean instead of an integer).
    % TODO: use symmetry trick by computing half of the matrix product? (row(i:c) * column(i:c) with i synchronized in both, thus that column cannot be of index lower than row).
    % TODO: reshuffle the adjacency matrix to get a band matrix? http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3148830/
    % TODO: use Locality of Reference to optimize operations?
    % TODO: use adjacency list instead of adjacency matrix?
    % TODO: use Modularity theory to efficiently use the clusters?
    % TODO: For tournament-based networks, it is also possible to use the non-symmetry by reordering the matrix: if a digraph is acyclic, then it is possible to order the points of D so that the adjacency matrix upper triangular (i.e., all positive entries are above the main diagonal).

    if ~silent; aux.printtime(toc()); end; % just to show performance, learning the network (adjacency matrix) _was_ the bottleneck

    if miterator > 0 % for debug cases, it may be nice to keep messages to compare with sparsemessages and network
        clear messages; % clear up some memory
    end
end
%network = network + network'; % TRICKS: using matrix symmetry to fill the remainder. NOTE: when using miterator, you need to fill the symmetry only at the end of the loop, else you will run out of memory
%network = logical(network); % = min(network,1) threshold because values can only be 1 at max (but if there were duplicate messages, some entries can have a higher value than 1)
%network = or(network, network'); % not necessary when using the vectorized version
% Note that  or(network, network') = min(network + network', 1) but the latter is a bit slower on big datasets
sparsemessages = logical(sparsemessages); % NOTE: prefer logical(x) rather than min(x, 1) because: logical is faster, and also the ending data will take less storage space (about half)

% Clean up memory, else MatLab/Octave keep everything in memory
if ~debug
    clear messages;
end

if ~silent; fprintf('-- Finished learning!\n'); aux.flushout(); end;

% count_cliques = sum(network^2 == c) % count the total number of cliques. WARNING: this is VERY slow but there's no better way to my knowledge



% == Compute density and some practical and theoretical stats
real_density = full(  (sum(sum(network)) - sum(diag(network))) / (Chi*(Chi-1) * l^2)  );
if ~silent
    fprintf('-- Computing density\n'); aux.flushout();
    real_density % density = (number_of_links - loops) / max_number_of_links; where max_number_of_links = (Chi*(Chi-1) * l^2).
    theoretical_average_density = 1 - (1 - c * (c-1) / (Chi * (Chi - 1) * l^2) )^m
    total_number_of_messages_really_stored = log2(1 - real_density) / log2(1 - c*(c-1) / (Chi*(Chi-1)*l.^2))
    number_of_nodes = n % total number of nodes (l * c)
    number_of_active_links = (nnz(network) - nnz(diag(network))) / 2 % total number of active links
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
