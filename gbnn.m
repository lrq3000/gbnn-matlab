function [network, real_density, error_rate] = gbnn(Xlearn, Xtest, ...
                                                                                  m, miterator, l, c, erasures, iterations, tampered_messages_per_test, tests, ...
                                                                                  Chi, guiding_mask, gamma_memory, threshold, propagation_rule, filtering_rule, tampering_type, ...
                                                                                  residual_memory, variable_length, concurrent_cliques, GWTA_first_iteration, GWTA_last_iteration, ...
                                                                                  silent, debug)
%
% function [network, real_density, error_rate] = gbnn(Xlearn, Xtest,
%                                                                                  m, miterator, l, c, erasures, iterations, tampered_messages_per_test, tests,
%                                                                                  Chi, guiding_mask, gamma_memory, threshold, propagation_rule, filtering_rule, tampering_type,
%                                                                                  residual_memory, variable_length, concurrent_cliques, GWTA_first_iteration, GWTA_last_iteration,
%                                                                                  silent, debug)
%
% #### Gripon-Berrou Neural Network, aka thrifty clique codes. Implementation in MatLab, by Vincent Gripon. Coded and tested on Octave 3.6.4 Windows.
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
%   Also you can place a keyboard command at the end of the script, call one run and then use full(reshape([init ; out ; propag], Chi*l, [])) to show the evolution of the decoding process.
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
if exist('Xlearn', 'var') && ~isempty(Xlearn)
    m = size(Xlearn, 1);
end
if exist('Xtest', 'var') && ~isempty(Xtest)
    tampered_messages_per_test = size(Xtest, 1);
end

if ~exist('gamma_memory', 'var') || isempty(gamma_memory)
    gamma_memory = 0;
end
if ~exist('threshold', 'var') || isempty(threshold)
    threshold = 0;
end
if ~exist('Chi', 'var') || isempty(Chi)
    Chi = c;
end
if ~exist('guiding_mask', 'var')
    guiding_mask = false;
end
if ~exist('propagation_rule', 'var') || ~ischar(propagation_rule)
    propagation_rule = 'sum';
end
if ~exist('filtering_rule', 'var') || ~ischar(filtering_rule)
    filtering_rule = 'wta';
end
if ~exist('tampering_type', 'var') || ~ischar(tampering_type)
    tampering_type = 'erase';
end

if ~exist('residual_memory', 'var')
    residual_memory = 0;
end
if ~exist('variable_length', 'var') || isempty(variable_length)
    variable_length = false;
end
if ~exist('concurrent_cliques', 'var') || isempty(concurrent_cliques)
    concurrent_cliques = 1; % 1 is disabled, > 1 enables and specify the number of concurrent messages/cliques to decode concurrently
end
if ~exist('GWTA_first_iteration', 'var') || isempty(GWTA_first_iteration)
    GWTA_first_iteration = false;
end
if ~exist('GWTA_last_iteration', 'var') || isempty(GWTA_last_iteration)
    GWTA_last_iteration = false;
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

    % -- Test variables
    erasures
    iterations
    tampered_messages_per_test
    tests

    % -- 2014 update
    gamma_memory
    threshold
    Chi
    guiding_mask
    propagation_rule
    filtering_rule
    tampering_type

    % -- Custom extensions
    residual_memory
    variable_length
    concurrent_cliques
    debug
end



% == Init data structures and other vars (need to do that before the miterator) - DO NOT TOUCH
sparse_cliques = true; % enable the creation of sparse cliques if Chi > c (cliques that don't use all available clusters but just c clusters per one message)
if Chi <= c
    Chi = c; % Chi can't be < c, thus here we ensure that
    sparse_cliques = false;
end
n = l * Chi; % total number of nodes
sparsemessages = logical(sparse(m,n)); % Init and converting to a binary sparse matrix
network = logical(sparse(n,n)); % init and converting to a binary sparse matrix

% Setup correct values for k (this is an automatic guess, but a manual value can be better depending on your dataset)
k = c*concurrent_cliques; % with propagation_rules GWTA and k-GWTA, usually we are looking to find at least as many winners as there are characters in the initial messages, which is at most c*concurrent_cliques (it can be less if the concurrent_cliques share some nodes, but this is unlikely if the density is low)
if strcmpi(filtering_rule, 'kWTA') || strcmpi(filtering_rule, 'kLKO') || strcmpi(filtering_rule, 'WsTA') % for all k local algorithms (k-WTA, k-LKO, WsTA, ...), k should be equal to the number of concurrent_cliques, since per cluster (remember that the rule here is local, thus per cluster) there is at most as many different characters per cluster as there are concurrent_cliques (since one clique can only use one node per cluster).
    k = concurrent_cliques;
end

mtest = m;
if ~isempty(Xtest)
    mtest = size(Xtest, 1);
end

if miterator > m
    miterator = 0;
end

% -- A few error checks
if erasures > c
    error('Erasures > c which is not possible');
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
    if ~isempty(Xlearn)
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
    offsets = 0:(l*Chi):(mgen*l*Chi);
    idxs = bsxfun(@plus, offsets(1:end-1)', idxs); % account for the row number now by generating a vector of index shift per row (eg: [0 lc 2lc 3lc 4lc ...]')
    idxs = idxs + ((M-1)*miterator)*Chi*l; % offset all indexes to the current miterator position (by just skipping previous messages rows)
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
    if M == 1 % case when network is empty, this is faster
        if aux.isOctave()
            network = logical(sparsemessages' * sparsemessages); % Credits go to Christophe for the tip!
        else % MatLab can't do matrix multiplication on logical (binary) matrices... must convert them to double beforehand
            dsparsemessages = double(sparsemessages);
            network = logical(dsparsemessages' * dsparsemessages);
        end
    else % case when we iteratively append new messages, we update the previous network
        if aux.isOctave()
            network = or(network, logical(sparsemessages' * sparsemessages)); % same as min(network + sparsemessages'*sparsemessages, 1)
        else
            dsparsemessages = double(sparsemessages);
            network = or(network, logical(dsparsemessages' * dsparsemessages)); % same as min(network + sparsemessages'*sparsemessages, 1)
        end
    end
    % Vectorized version 2 draft: use directly the indices without matrix multiplication to avoid useless computations because of symmetry: vectorized_indices = reshape(mod(find(sparsemessages'), l*Chi), c, m)'
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
if ~silent;
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
    spurious_cliques_proba = 1 - (1 - real_density^(c-erasures))^(erasures*(l-1)+l*(Chi-c)) % = theoretical_error_rate, spurious cliques = nonvalid cliques that we did not memorize and which rests inopportunely on the edges of valid cliques, which we learned and want to remember. In other words: what is the probability of emergence of wrong cliques that we did not learn but which emerges from combinations of cliques we learned? This is influenced heavily by the density (higher density = more errors)
    theoretical_error_correction_proba = 1 - spurious_cliques_proba
end;


% #### Prediction phase
% == Run the test (error correction) and compute error rate (in reconstruction of original message without erasures)
if ~silent; fprintf('#### Testing phase (error correction of tampered messages with %i erasures)\n', erasures); aux.flushout(); end;
sparsemessagestest = sparsemessages; % by default, use to test the same set as for learning
if ~isempty(Xtest); sparsemessagestest = Xtest; end; % else if defined, use the test set
err = 0; % error score
parfor t=1:tests
    if ~silent
        if tests < 20 || mod(tests, t) == 0
            fprintf('== Running test %i with %i tampered messages\n', t, tampered_messages_per_test); aux.flushout();
        end
    end

    % -- Generation of a tampered message to remember
    % (with erasure of a random character in the message)
    %if ~silent; fprintf('-- Generating tampered messages\n'); aux.flushout(); end;
    if ~silent; tic(); end;

    % 1- select a list of random messages
    mconcat = 1; % by default there's no concurrent clique
    if concurrent_cliques > 1; mconcat = concurrent_cliques; end; % if concurrent_cliques is enabled, we have to generate more messages (as many as concurrent_cliques requires)

    % Generate random indices to randomly choose messages
    %rndidx = unidrnd(mtest, [mconcat tampered_messages_per_test]); % TRICKS: unidrnd(m, [SZ]) is twice as fast as unidrnd(m, dim1, dim2)
    rndidx = randi([1 mtest], mconcat, tampered_messages_per_test);

    % Fetch the random messages from the generated indices
    input = sparsemessagestest(rndidx,:)'; % just fetch the messages and transpose them so that we have one sparsemessage per column (we don't generate them this way even if it's possible because of optimization: any, or, and sum are more efficient column-wise than row-wise, as any other MatLab/Octave function).
    init = input; % backup the original message before tampering, we will use the original to check if the network correctly corrected the erasure(s)

    %if ~debug; clear rndidx; end; % clear up memory

    % 2- randomly tamper them (erasure or noisy bit-flipping of a few characters)
    % -- Random erasure of random active characters (which the network is more tolerant than noise, which is normal and described in modern error correction theory)
    if strcmpi(tampering_type, 'erase')
        % The idea is that we will randomly pick several characters to erase per message (by extracting all nonzeros indices, order per-column/message, and then shuffling them to finally select only a few indices per column to point to the characters we will erase)
        [~, idxs] = sort(input, 'descend'); % sort the messages to get the indices of the nonzeros values, but still organized per-column (which find() doesn't provide)
        idxs = idxs(1:c, :); % memory efficiency: trim out indices of the zero values (since we are sure that at most a message contains c characters)
        idxs = aux.shake(idxs); % per-column shuffle indices! This is how we randomly pick characters.
        idxs = idxs(1:erasures, :); % select the number of erasures we want
        idxs = bsxfun(@plus, idxs, 0:l*Chi:l*Chi*(tampered_messages_per_test*concurrent_cliques-1) ); % offset indices to take account of the column (since sort resets indices count per column)
        idxs(idxs == 0) = []; % remove non valid indices (if variable_length, some characters may have less than the number of characters we want to erase) TODO: ensure that a variable_length message keeps at least 2 nodes
        input(idxs(1:erasures, :)) = 0; % erase those characters
    % -- Random noise (bit-flipping of randomly selected characters)
    elseif strcmpi(tampering_type, 'noise')
        % The idea is simple: we generate random indices to be "noised" and we bit-flip them using modulo.
        %idxs = unidrnd(Chi*l, [erasures mconcat*tampered_messages_per_test]); % generate random indices to be tampered
        idxs = unidrnd([1 Chi*l], erasures, mconcat*tampered_messages_per_test); % generate random indices to be tampered
        idxs = bsxfun(@plus, idxs, 0:l*Chi:l*Chi*(tampered_messages_per_test*concurrent_cliques-1) ); % offset indices to take account of the column = message (since sort resets indices count per column)
        input(idxs) = mod(input(idxs) + 1, 2); % bit-flipping! simply add one to all those entries and modulo one, this will effectively bit-flip them.
    % Else error, the tampering_type does not exist
        else
            error('Unrecognized tampering_type: %s', tampering_type);
    end

% OLD UNGUARANTEED METHOD if sparse_cliques is enabled
%    parfor j=0:tampered_messages_per_test:tampered_messages_per_test*(concurrent_cliques-1)
%        indexes = randperm(c); % generate a random permutation so that we are sure that each time we pick a character it wasn't already erased before
%        parfor i=1:erasures % execute in parallel since we work on different parts of the message (guaranteed because we use randperm)
%            charstart = (indexes(i)-1)*l+1; % start index of the thrifty code for this character
%            charend = indexes(i)*l; % end index of the thrifty code
%            input(charstart:charend, j+1:j+tampered_messages_per_test) = 0; % for each character (in a random order), we tamper it (precisely we tamper the thrifty code, hence why we copy a vector of zeros)
%        end
%    end

    % If concurrent_cliques is enabled, we must mix up the messages together after we've done the characters erasure
    if concurrent_cliques > 1
        % Mix up init
        init = reshape(init, l*Chi*tampered_messages_per_test, concurrent_cliques)';
        init = any(init); % mix up messages (by stacking concurrent_cliques messages side-by-side and then summing/anying them)
        %init = reshape(init, tampered_messages_per_test, l*Chi)'; % WRONG % unstack the messages vector into a matrix with one mixed sparsemessage per column
        init = reshape(init', l*Chi, tampered_messages_per_test); % unstack the messages vector into a matrix with one mixed sparsemessage per column

        % Mix up the tampered messages
        input = reshape(input, l*Chi*tampered_messages_per_test, concurrent_cliques)';
        input = any(input);
        input = reshape(input', l*Chi, tampered_messages_per_test);
    end

    if ~silent; aux.printtime(toc()); end;

    if ~silent; tperf = cputime(); end;
    % -- Prediction step: feed the tampered message to the network and wait for it to converge to a stable state, hopefully the corrected message.
    %if ~silent; fprintf('-- Feed to network and wait for convergence\n'); aux.flushout(); end;
    if guiding_mask % if enabled, prepare the guiding mask (the list of clusters that we will keep, all the other nodes from other clusters will be set to 0). This guiding mask can be defined manually if you want, here to do it automatically we compute it from the initial untampered messages, thus we will keep nodes activated only in the clusters where there were activated nodes in the initial message.
        gmask = any(reshape(init, l, tampered_messages_per_test * Chi)); % any is better than sum in our case, and it's also faster and keeps the logical datatype!
    end
    for iter=1:iterations % To let the network converge towards a stable state...
        if ~silent
            fprintf('-- Propagation iteration %i\n', iter); aux.flushout();
            tic();
        end

        % 1- Update the network's state: Push message and propagate through the network
        % NOTE: this is the CPU bottleneck if you use many messages with c > 8

        % -- Vectorized version - fastest and low memory overhead
        % Sum-of-Sum: we simply compute per node the sum of all incoming activated edges
        if strcmpi(propagation_rule, 'sum')
            % We use the standard way to compute the propagation in an adjacency matrix: by matrix multiplication
            % Sum-of-Sum / Matrix multiplication is the same as computing the in-degree(a) for each node a, since each connected and activated link to a is equivalent to +1, thus this is equivalent to the total number of connected and activated link to a.
            if aux.isOctave()
                propag = network * input; % Propagate the previous message state by using a simple matrix product (equivalent to similarity/likelihood? or is it more like a markov chain convergence?). Eg: if network = [0 1 0 ; 1 0 0], input = [1 1 0] will match quite well: [1 1 0] while input = [0 0 1] will not match at all: [0 0 0]
            else % MatLab cannot do matrix multiplication on logical matrices...
                propag = double(network) * double(input);
            end
        % Else error, the propagation_rule does not exist
        else
            error('Unrecognized propagation_rule: %s', propagation_rule);
        % TODO: sum-of-max use spfunc() or arrayfun() to do a custom matrix computation per column: first an and, then a reshape to get only nodes per cluster on one colum, then any, then we have our new message! Or compute sos then diff(sos - som) then sos - diff(sos-sum)
        % TODO: Sum-of-Max = for each node, compute the sum of incoming link per cluster (thus if a node is connected to 4 other nodes, but 3 of them belong to the same cluster, the score will be 2 = 1 for the node alone in its cluster + 1 for the other 3 belonging to the same cluster).
        % TODO: Normalized Sum-of-Sum = for each node, compute the sum of sum but divide the weight=1 of each incoming link by the number of activated node in the cluster where the link points to (thus if a node is connected to 4 other nodes, but 3 of them belong to the same cluster, the score will be 1 + 1/3 + 1/3 + 1/3 = 2). The score will be different than Sum of Max if concurrent_cliques is enabled (because then the number of activated nodes can be higher and divide more the incoming scores).
        end
        if gamma_memory > 0
            propag = propag + (gamma_memory .* input); % memory effect: keep a bit of the previous nodes scores
        end
        if threshold > 0 % activation threshold: set to 0 all nodes with a score lower than the activation threshold (ie: those nodes won't emit a spike)
            propag(propag < threshold) = 0;
        end;

        % -- Semi-vectorized version 1 - fast but very memory consuming
        % Using the adjacency matrix as an adjacency list of neighbors per node: we just fetch the neighbors indices and then compute the final message
%        propag_idx = mod(find(input)-1, c*l)+1; % get the indices of the currently activated nodes (we use mod because we compute at once the indices for all messages, and the indices will be off starting from the second message)
%        activated_neighbors = network(:, propag_idx); % propagation: find the list of neighbors for each currently activated node = just use the current nodes indices in the adjacency list. Thus we get one list of activated neighbors per currently activated node.
%        messages_idx = [0 cumsum(sum(input))]; % now we need to unstack all lists of neighbors per message instead of per node (we will get one list of neighbor per activated neuron in one message, thus we need to unstack the matrix into submatrixes: one per message)
%        propag = sparse(c*l, tampered_messages_per_test); % preallocate the propag matrix
%        parfor i=1:tampered_messages_per_test % now we will sum all neighbors activations into one message (because we have many submatrices = messages containing lists of activated neighbors per currently activated node, but we want one list of activated neighbors for all nodes = one message)
%            propag(:,i) = sum( activated_neighbors(:,messages_idx(i)+1:messages_idx(i+1)) , 2); % TODO: Vectorize this! find a way to do it without a for loop!
%        end

        % -- Semi-vectorized version 2 - slower but easy on memory
        % same as version 1 but to lower memory footprint we compute the activated_neighbors in the loop instead of outside.
%        propag_idx = mod(find(input)-1, c*l)+1;
%        cluster_idx = [0 cumsum(sum(input))];
%        propag = sparse(c*l, tampered_messages_per_test);
%        parfor i=1:tampered_messages_per_test
%            propagidx_per_cluster = propag_idx(cluster_idx(i)+1:cluster_idx(i+1));
%            activated_neighbors = network(:, propagidx_per_cluster);
%            propag(:,i) = sum( activated_neighbors , 2);
%        end


        % 2- Winner-takes-all filtering (shutdown all the other nodes, we keep only the most likely characters)
        % -- Semi-vectorized version
        out = logical(sparse(size(input,1), size(input,2))); % empty binary sparse matrix, it will later store the next network state after winner-takes-all is applied
        % For each cluster(=character position), we select winners and shutdown the others (select the character that has max score at this position of the message)
%        parfor i=1:c % execute in parallel, since winner-takes-all is a parallel algorithm. But maybe we can vectorize this process?
%           charstart = (i-1)*l+1;
%           charend = i*l;
%            winner_value = max(propag(charstart:charend, :), [], 1); % what is the maximum output value (considering all the nodes in this character)
%           if tampered_messages_per_test > 1
%               out(charstart:charend, :) = bsxfun(@eq, propag(charstart:charend, :), winner_value); % Winner-takes-all: keep (set to 1) all nodes whose attained the maximum output value/score (= number of in-edges from activated nodes) and shutdown the others (set 0), and concatenate into the next message state "out"
%           else
%               out(charstart:charend, :) = (propag(charstart:charend) == winner_value); % more efficient (twice the speed) if only one message is to be tested
%           end
%        end

        % -- Vectorized version - fastest!
        % Winner-take-all : per cluster, keep only the maximum score node active (if multiple nodes share the max score, we keep them all activated). Thus the WTA is based on score value, contrarywise to k-WTA which is based on the number of active node k.
        if strcmpi(filtering_rule, 'wta')
            % The idea is that we will break the clusters and stack them along as a single long cluster spanning several messages, so that we can do a WTA in one pass (with a single max), and then we will unstack them back to their correct places in the messages
            propag = reshape(propag, l, tampered_messages_per_test * Chi); % reshape so that we can do the WTA by a simple column-wise WTA (and it's efficient in MatLab since matrices - and even more with sparse matrices - are stored as column vectors, thus it efficiently use the memory cache since this is the most limiting factor above CPU power). See also: Locality of reference.
            winner_value = max(propag); % what is the maximum output value (considering all the nodes in this character)
            if ~aux.isOctave()
                out = logical(bsxfun(@eq, propag, winner_value)); % Get the winners by filtering out nodes that aren't equal to the max score. Use bsxfun(@eq) instead of ismember.
            else
                out = logical(sparse(bsxfun(@eq, propag, winner_value))); % Octave does not overload bsxfun with sparse by default, so bsxfun removes the sparsity, thus the propagation at iteration > 1 will be horribly slow! Thus we have to put it back manually. MatLab DOES overload bsxfun with sparsity so this only applies to Octave. See: http://octave.1599824.n4.nabble.com/bsxfun-and-sparse-matrices-in-Matlab-td3867746.html
            end
            out = and(propag, out); % IMPORTANT NO FALSE WINNER TRICK: if sparse_cliques or variable_length, we deactivate winning nodes that have 0 score (but were chosen because there's no other activated node in this cluster, and max will always return the id of one node! Thus we have to compare with the original propagation matrix and see if the node was really activated before being selected as a winner).
            out = reshape(out, l*Chi, tampered_messages_per_test);
        % k-Winner-take-all : keep the best first k nodes having the maximum score, over the whole message. This is kind of a cheat because we must know the original length of the messages, the variable k is a way to tell the algorithm some information we have to help convergence
        elseif strcmpi(filtering_rule, 'kwta')
            propag = reshape(propag, l, tampered_messages_per_test * Chi); % reshape so that we can do the WTA by a simple column-wise WTA (and it's efficient in MatLab since matrices - and even more with sparse matrices - are stored as column vectors, thus it efficiently use the memory cache since this is the most limiting factor above CPU power). See also: Locality of reference.
            [~, idxs] = sort(propag, 'descend');
            idxs = bsxfun( @plus, idxs, 0:l:((tampered_messages_per_test * Chi * l)-1) );
            [I, J] = ind2sub(size(propag), idxs(1:k, :));
            out = logical(sparse(I, J, 1, l, tampered_messages_per_test * Chi));
            out = and(propag, out); % IMPORTANT NO FALSE WINNER TRICK: if sparse_cliques or variable_length, filter out winning nodes with 0 score (selected because there's no other node with any score in this cluster, see above in filtering_rule = 0)
            out = reshape(out, l*Chi, tampered_messages_per_test);
        % One Global Winner-take-all: only keep one value, but at the last iteration keep them all
        elseif strcmpi(filtering_rule, 'ogwta') && iter < iterations
            [~, winner_idxs] = max(propag); % get indices of the max score for each message
            winner_idxs = bsxfun(@plus, winner_idxs,  0:l*Chi:l*Chi*(tampered_messages_per_test-1)); % indices returned by max are per-column, here we add the column size to offset the indices, so that we get true MatLab indices
            winner_idxs = winner_idxs(propag(winner_idxs) > 0); % No false winner trick: we check the values returned by the k best indices in propag, and keep only indices which are linked to a non null value.
            if nnz(winner_idxs) > 0 % If there is at least one non null value (there's at least one remaining index after the no false winner trick)
                [I, J] = ind2sub(size(propag), winner_idxs); % extract the sparse matrix non null values indices
                out = logical(sparse(I, J, 1, l*Chi, tampered_messages_per_test)); % create the logical sparse matrix
            else % Else there's not even one non null value, this means that all messages have 0 max score (the network shut down), then we just have to create an empty output
                out = logical(sparse(l*Chi, tampered_messages_per_test)); % create an empty logical sparse matrix
            end
        % Global Winner-take-all: keep only nodes which have maximum score over the whole message. Same as WTA but instead of doing inhibition per-cluster, it's per the whole message/network.
        elseif strcmpi(filtering_rule, 'gwta') || (strcmpi(filtering_rule, 'ogwta') && iter == iterations) || (GWTA_first_iteration && iter == 1) || (GWTA_last_iteration && iter == iterations)
            winner_vals = max(propag); % get global max scores for each message
            if ~aux.isOctave()
                out = logical(bsxfun(@eq, winner_vals, propag));
            else
                out = logical(sparse(bsxfun(@eq, winner_vals, propag)));
            end
            out = and(propag, out); % No false winner trick
        % Global k-Winners-take-all: keep the best k first nodes having the maximum score over the whole message (same as k-WTA but at the message level instead of per-cluster).
        elseif strcmpi(filtering_rule, 'GkWTA')
            % Instead of removing indices of scores that are below the k-max scores, we will rather find the indices of these k-max scores and recreate a logical sparse matrix from scratch, this is a lot faster and memory efficient
            [~, idxs] = sort(propag, 'descend'); % Sort scores with best scores first
            idxs = idxs(1:k,:); % extract the k best scores indices (for propag)
            idxs = bsxfun(@plus, idxs,  0:l*Chi:l*Chi*(tampered_messages_per_test-1)); % indices returned by sort are per-column, here we add the column size to offset the indices, so that we get true MatLab indices
            winner_idxs = idxs(propag(idxs) > 0); % No false winner trick: we check the values returned by the k best indices in propag, and keep only indices which are linked to a non null value.
            [I, J] = ind2sub(size(propag), winner_idxs); % extract the necessary I and J (row and columns) indices vectors to create a sparse matrix
            out = logical(sparse(I, J, 1, l*Chi, tampered_messages_per_test)); % finally create a logical sparse matrix
        % WinnerS-take-all: per cluster, select the kth best score, and accept all nodes with a score greater or equal to this score - similar to kWTA but all nodes with the kth score are accepted, not only a fixed number k of nodes
        % Global WinnerS-take-all: same as WsTA but per the whole message
        % Both are implemented the same way, the only difference is that for global we process winners per message, and with local we process them per cluster
        elseif strcmpi(filtering_rule, 'WsTA') || strcmpi(filtering_rule, 'GWsTA')
            % Local WinnerS-take-all: Reshape to get only one cluster per column (instead of one message per column)
            if strcmpi(filtering_rule, 'WsTA')
                propag = reshape(propag, l, tampered_messages_per_test * Chi);
            end

            max_scores = sort(propag,'descend');
            kmax_score = max_scores(k, :);
            kmax_score(kmax_score == 0) = realmin(); % No false winner trick: better version of the no false winner trick because we replace 0 winner scores by the minimum real value, thus after we will still get a sparse matrix (else if we do the trick only after the bsxfun, we will get a matrix filled by 1 where there are 0 and the winner score is 0, which is a lot of memory used for nothing).
            out = logical(bsxfun(@ge, propag, kmax_score));
            if aux.isOctave(); out = sparse(out); end; % Octave's bsxfun breaks the sparsity...
            out = and(propag, out); % No false winner trick: avoids that if the kth max score is in fact 0, we choose 0 as activating score (0 scoring nodes will be activated, which is completely wrong!). Here we check against the original propagation matrix: if the node wasn't activated then, it should be now.

            % Local WinnerS-take-all: Reshape back the clusters into messages
            if strcmpi(filtering_rule, 'WsTA')
                out = reshape(out, l*Chi, tampered_messages_per_test);
            end

        % Loser-kicked-out (locally, per cluster, we kick loser nodes with min score except if min == max of this cluster).
        % Global Loser-Kicked-Out (deactivate all nodes with min score in the message)
        % Both are implemented the same way, the only difference is that for global we process losers per message, and with local we process them per cluster
        elseif strcmpi(filtering_rule, 'LKO') || strcmpi(filtering_rule, 'GLKO')
            % Local Losers-Kicked-Out: Reshape to get only one cluster per column (instead of one message per column)
            if strcmpi(filtering_rule, 'LKO')
                propag = reshape(propag, l, tampered_messages_per_test * Chi);
            end

            % get sorted indices (with 0's at the end)
            sorted = sort(propag);

            % Re-Sort in descending way but use the previous ascended sort to place 0 at the end, while updating idxs to reflect the resorting, so that we still get the correct indexing for propag (everything is computed relatively to propag)
            out = sort(propag, 'descend'); % resort by descending order (so the max is first and zeros at the end)
            [I, J] = find(out ~= 0); % find everwhere there is a non-zero
            out(out ~= 0) = sorted(sorted ~= 0); % get the resorted scores matrix by resorting the non-zeros in an ascending way (but only those, we don't care about zeros!)

            % Get the losers (min scoring nodes)
            losers_vals = out(1,:);

            % No false loser trick: remove losers that in fact have the maximum score
            % compare the losers scores against max score and find indices where losers scores equal max scores (per cluster)
            false_losers = (losers_vals == max(propag)); % find all losers scores that are equal to the maximum of the cluster
            false_losers = and(false_losers, losers_vals); % trim out null scores (where there is in fact no activated node, for example a sparse cluster, represented here by a column filled with 0s in losers_vals or propag)
            false_losers = find(false_losers); % get indices of the false losers
            if nnz(false_losers) > 0 % if there is at least one false loser
                losers_vals(false_losers) = -1; % in-place remove the false losers: A(idxs) = [] allows to remove an entry altogether
            end

            % Finally, deactivate losers from the propag matrix, this will be our new out messages matrix
            out = logical(propag);
            if ~aux.isOctave()
                out(bsxfun(@eq, propag, losers_vals)) = 0;
            else
                out(logical(sparse(bsxfun(@eq, propag, losers_vals)))) = 0;
            end

            % Local Losers-Kicked-Out: Reshape back the clusters into messages
            if strcmpi(filtering_rule, 'LKO')
                out = reshape(out, l*Chi, tampered_messages_per_test);
            end
        % k-Losers-kicked-out (locally, per cluster, we kick exactly k loser nodes with min score except if min == max of this cluster).
        % Global-k-Losers-Kicked-Out: deactivate k nodes with the minimum score, per the whole message
        % Optimal/One Loser-Kicked-Out (only one node with min score is kicked) and Optimal Global Loser-Kicked-Out
        % Both are implemented the same way, the only difference is that for global we process losers per message, and with local we process per cluster
        elseif strcmpi(filtering_rule, 'kLKO') || strcmpi(filtering_rule, 'GkLKO') || ...
                    strcmpi(filtering_rule, 'oLKO') || strcmpi(filtering_rule, 'oGLKO')
            %[~, loser_idx] = min(propag(propag > 0)); % propag(propag > 0) is faster than nonzeros(propag)
            %propag(loser_idx) = 0;

            %loser_val = min(propag(propag > 0));
            %out = logical(propag);
            %out(find(propag == loser_val)(1)) = 0;

            % Local k-Losers-Kicked-Out: Reshape to get only one cluster per column (instead of one message per column)
            if strcmpi(filtering_rule, 'kLKO') || strcmpi(filtering_rule, 'oLKO')
                propag = reshape(propag, l, tampered_messages_per_test * Chi);
            end

            % get sorted indices
            [sorted, idxs] = sort(propag);
            idxs = bsxfun(@plus, idxs, 0:size(propag,1):(numel(propag)-size(propag,1))); % offset to get real indices since sort returns per-column indices (reset to 0 for each column)

            % Re-Sort to place 0 at the end, while updating idxs to reflect the resorting, so that we still get the correct indexing for propag (everything is computed relatively to propag)
            out = sort(propag, 'descend'); % resort by descending order (so the max is first and zeros at the end)
            [I, J] = find(out ~= 0); % find everwhere there is a non-zero
            losers_idxs = sparse(I, J, 1, size(propag,1), size(propag,2)); % create a new indices matrix with 1's where the non-zeros will be
            %out(out ~= 0) = sorted(sorted ~= 0); % get the resorted scores matrix
            losers_idxs(losers_idxs ~= 0) = idxs(sorted ~= 0); % get the final resorted indices matrix, by replacing the sparse non-zeros with the non-zeros from the first sort (first sort was with ascending order but there were 0 at the beginning, so now we don't care about 0, we just overwrite the values with the same ones but in ascending order).
            if strcmpi(filtering_rule, 'oLKO') || strcmpi(filtering_rule, 'oGLKO')
                losers_idxs = losers_idxs(1, :); % special case: oLKO = kLKO with k = 1 exactly
            else
                losers_idxs = losers_idxs(1:k, :); % keep only the indices of the k-worst scores (k min scores)
            end

            % No false loser trick: remove losers that in fact have the maximum score
            winners_vals = max(propag); % get max score per cluster
            % fetch k-losers scores (values) per cluster by using the indices
            losers_vals = losers_idxs;
            losers_vals(losers_vals ~= 0) = propag(losers_idxs(losers_idxs ~= 0)); % this is the trick: we keep the shape of the losers_idxs matrix but we replace the non zeros values by the scores from propag
            % compare the losers scores against max score and find indices where losers scores equal max scores (per cluster)
            false_losers = bsxfun(@eq, losers_vals, winners_vals); % find all losers scores that are equal to the maximum of the cluster
            false_losers = and(false_losers, losers_vals); % trim out null scores (where there is in fact no activated node, for example a sparse cluster, represented here by a column filled with 0s in losers_vals or propag)
            false_losers = find(false_losers); % get indices of the false losers
            if nnz(false_losers) > 0 % if there is at least one false loser
                losers_idxs(false_losers) = []; % in-place remove the false losers: A(idxs) = [] allows to remove an entry altogether
            end

            % Finally, deactivate losers from the propag matrix, this will be our new out messages matrix
            out = logical(propag);
            losers_idxs(losers_idxs == 0) = []; % make sure that there's no empty indices by removing them (for sparse cliques, where there are some cliques where there's absolutely no character)
            if nnz(losers_idxs) > 0 % check that there's still at least one index to set to 0
                out(losers_idxs) = 0;
            end

            % Local k-Losers-Kicked-Out: Reshape back the clusters into messages
            if strcmpi(filtering_rule, 'kLKO') || strcmpi(filtering_rule, 'oLKO')
                out = reshape(out, l*Chi, tampered_messages_per_test);
            end
        % Concurrent Global k-Winners-take-all
        % CkGWTA = kGWTA + trimming out all scores that are below the k-th max score (so that if there are shared nodes between multiple messages, we will get less than c active nodes at the end)
        elseif strcmpi(filtering_rule, 'CGkWTA')
            % 1- kGWTA
            [~, idxs] = sort(propag, 'descend'); % Sort scores with best scores first
            idxs = idxs(1:k,:); % extract the k best scores indices (for propag)
            idxs = bsxfun(@plus, idxs,  0:l*Chi:l*Chi*(tampered_messages_per_test-1)); % indices returned by sort are per-column, here we add the column size to offset the indices, so that we get true MatLab indices
            winner_idxs = idxs(propag(idxs) > 0); % No false winner trick: we check the values returned by the k best indices in propag, and keep only indices which are linked to a non null value.
            [I, J] = ind2sub(size(propag), winner_idxs); % extract the necessary I and J (row and columns) indices vectors to create a sparse matrix
            temp = propag(idxs); % MatLab compatibility...
            propag2 = sparse(I, J, temp(temp > 0), l*Chi, tampered_messages_per_test); % finally create a sparse matrix, but contrarywise to the original implementation of kGWTA, we keep the full scores here, we don't turn it yet into binary, so that we can continue processing in the second stage.

            % 2- Trimming below k-th max scores
            temp = propag2;
            maxscores = max(temp);
            for i=1:concurrent_cliques-1
                if aux.isOctave()
                    temp = sparse(bsxfun(@mod, full(temp), maxscores));
                else
                    temp = bsxfun(@mod, temp, maxscores);
                end
                maxscores = max(temp);
            end
            clear temp;
            maxscores(maxscores == 0) = 1;
            if aux.isOctave()
                out = logical(sparse(bsxfun(@ge, propag2, maxscores)));
            else
                out = logical(bsxfun(@ge, propag2, maxscores));
            end
        % Else error, the filtering_rule does not exist
        else
            error('Unrecognized filtering_rule: %s', filtering_rule);
        end

        % 3- Some post-processing
        % Residual memory
        if residual_memory > 0
            out = out + (residual_memory .* input); % residual memory: previously activated nodes lingers a bit (a fraction of their activation score persists) and participate in the next iteration
        end

        % Guiding mask
        if guiding_mask % Apply guiding mask to filter out useless clusters. TODO: the filtering should be directly inside the propagation rule at the moment of the matrix product to avoid useless computations (but this is difficult to do this in MatLab in a vectorized way...).
            out = reshape(out, l, tampered_messages_per_test * Chi);
            if ~aux.isOctave()
                out = bsxfun(@and, out, gmask);
            else
                out = logical(sparse(bsxfun(@and, out, gmask)));
            end
            out = reshape(out, l*Chi, tampered_messages_per_test);
        end

        input = out; % set next messages state as current
        if ~silent; aux.printtime(toc()); end;
    end
    if ~silent
        fprintf('-- Propagation done!\n'); aux.flushout();
        aux.printcputime(cputime() - tperf, 'Propagation total elapsed cpu time is %g seconds.\n'); aux.flushout();
    end

    % -- Test score: compare the corrected message by the network with the original untampered message and check whether it's the same or not (partial or non correction are scored the same: no score)
    % If tampered_messages_per_test > 1, then the score is incremented per each unrecovered message, not per the whole pack
    %if ~silent; fprintf('-- Converged! Now computing score\n'); aux.flushout(); end;
    if ~silent; tic(); end;
    if residual_memory > 0 % if residual memory is enabled, we need to make sure that values are binary at the end, not just near-binary (eg: values of nodes with 0.000001 instead of just 0 due to the memory addition), else the error can't be computed correctly since some activated nodes will still linger after the last WTA!
        input = max(round(input), 0);
    end
    if tampered_messages_per_test > 1
        %err = err + sum(min(sum((init ~= input), 1), 1)); % this is a LOT faster than isequal() !
        err = err + nnz(sum((init ~= input), 1)); % even faster!
    else
        %err = err + ~isequal(init,input);
        err = err + any(init ~= input); % remove the useless sum(min()) when we only have one message to compute the error from, this cuts the time by almost half
    end
    if ~silent; aux.printtime(toc()); end;

end

% Finally, show the error rate and some stats
error_rate = err / (tests * tampered_messages_per_test);
if ~silent
    error_rate
    total_tampered_messages_tested = tests * tampered_messages_per_test
    aux.printcputime(cputime - totalperf, 'Total elapsed cpu time is %g seconds.\n'); aux.flushout();
end

end % endfunction
