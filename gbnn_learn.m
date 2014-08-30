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
%- m : there are 3 possible types of values: number of messages or a matrix of messages (composed of numbers ranging from 1 to l and of length/columns c per row) or a target density (this will automatically convert to a number of messages).
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
    ... % Overlays / Tags extension
    'enable_overlays', false, ...
    'overlays_max', 0, ...
    'overlays_interpolation', 'norm', ...
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
if isscalar(m) && m > 0 && m < 1 % If user provided a density, we convert to an integer number of messages to learn
    cnetwork_stats = gbnn_theoretical_stats('Chi', Chi, 'c', c, 'l', l, 'd', m);
    m = round(cnetwork_stats.M);
    clear cnetwork_stats;
    if m < 0; error('The density provided in argument m is too small to generate even one message!'); end;
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
if ~exist('cnetwork', 'var') || isempty(cnetwork) || isempty(cnetwork.primary) % reuse network if provided
    cnetwork = struct();
    cnetwork.primary = struct( ...
        'net', logical(sparse(n,n)), ... % preallocating and converting to a binary sparse matrix
        'args', struct( ...
            'l', l, ...
            'c', c, ...
            'Chi', Chi, ...
            'n', n, ...
            'sparse_cliques', sparse_cliques) ...
        );
    if enable_overlays; cnetwork.primary.net = double(cnetwork.primary.net); end; % if we will compute overlays, it's useless to have a logical matrix since it will contain integers
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

    % Moved to gbnn_messages2thrifty.m
    thriftymessages = or(thriftymessages, gbnn_messages2thrifty(messages, l, miterator, M, m));

    if ~silent; aux.printtime(toc()); end; % For perfs


    % == Create network = learn the network
    if ~silent
        fprintf('-- Learning messages into the network\n'); aux.flushout();
        tic();
    end

    % Moved to gbnn_construct_network.m
    if M == 1 && ~networkprovided % case when network is empty, this is faster
        if ~enable_overlays % Standard case: a simple matrix multiplication to create an adjacency matrix
            cnetwork.primary.net = gbnn_construct_network(thriftymessages);
        else % Overlays/Tags case: a generalized matrix multiplication max-of-products to assign to each edge the tag id of the latest message learned (instead of always having a tag of 1)
            overlays_range = (1:m)'; % Assign a unique overlay id to each message. We will reduce the number of overlays/tags later at correction/prediction step, so that we can learn only one network and try several different number of tags or reduction methods without having to relearn another network.
            %if overlays_max > 1 % if you would rather reduce the number of overlays directly at the learning step instead of correct step, you can do it here like this.
                %if strcmpi(overlays_interpolation, 'mod')
                    %overlays_range = mod(overlays_range-1, 3)+1;
                %end
            %end

            % -- Vectorized version, slower but more mathematically justified
            %outop = @max;
            %if size(thriftymessages, 1) == 1; outop = @(x) max(x, [], 1); end;
            %cnetwork.primary.net = gbnn_construct_network(bsxfun(@times, double(thriftymessages), overlays_range), double(thriftymessages), outop, @times); % Overlays computation = Generalized matrix multiplication, with max-of-products instead of sum-of-products (so that for each edge we keep the tag id of the latest/most recent message learned, in the order of the messages stack).

            % -- Loop semi-vectorized indexing version, a lot faster if you have JIT (based on Ehsan's method for learning) - the result is equivalent to the vectorized version
            for mi = 1:m % loop for each message
                msgL = find(thriftymessages(mi, :)) ; % pick up a message
                cnetwork.primary.net(msgL , msgL) = mi ; % Push the message to learn by index and assign the message id as the edges tags
            end
        end
    else % case when we iteratively append new messages (either because of miterator or because user provided a network to reuse), we update the previous network
        if ~enable_overlays % Standard case: a simple matrix multiplication to create an adjacency matrix
            cnetwork.primary.net = or(cnetwork.primary.net, gbnn_construct_network(thriftymessages)); % use a or() to iteratively append new edges over the old ones
        else % Overlays/Tags case
            prev_max_overlay = max(cnetwork.primary.net(:)); % Assign a unique overlay id to each message, same as above...

            % -- Vectorized version, slower but more mathematically justified
            %overlays_range = (1+prev_max_overlay:m+prev_max_overlay)'; % Offset because of the miterator
            %outop = @max;
            %if size(thriftymessages, 1) == 1; outop = @(x) max(x, [], 1); end;
            %cnetwork.primary.net = max(cnetwork.primary.net, gbnn_construct_network(bsxfun(@times, double(thriftymessages), overlays_range), double(thriftymessages), outop, @times)); % same as above...

            % -- Loop semi-vectorized indexing version, a lot faster if you have JIT (based on Ehsan's method for learning) - the result is equivalent to the vectorized version
            for mi = 1:m % loop for each message
                msgL = find(thriftymessages(mi, :)) ; % pick up a message
                cnetwork.primary.net(msgL , msgL) = mi + prev_max_overlay ; % Push the message to learn by index and assign the message id as the edges tags
            end
        end
    end

    % Attach overlays arguments in the network structure
    if enable_overlays
        cnetwork.primary.args.overlays_max = overlays_max;
        cnetwork.primary.args.overlays_interpolation = overlays_interpolation;
    end

    if ~silent; aux.printtime(toc()); end; % just to show performance, learning the network (adjacency matrix) _was_ the bottleneck

    if miterator > 0 % for debug cases, it may be nice to keep messages to compare with thriftymessages and network
        clear messages; % clear up some memory
    end
end
thriftymessages = logical(thriftymessages); % NOTE: prefer logical(x) rather than min(x, 1) because: logical is faster, and also the ending data will take less storage space (about half)

% Clean up memory, else MatLab/Octave keep everything in memory
clear messages;

if ~silent; fprintf('-- Finished learning!\n'); aux.flushout(); end;

% count_cliques = sum(cnetwork^2 == c) % count the total number of cliques. WARNING: this is VERY slow but there's no better way to my knowledge



% == Compute density and some practical and theoretical stats
real_density = full(  (nnz(cnetwork.primary.net) - nnz(diag(cnetwork.primary.net))) / (Chi*(Chi-1) * l^2)  );
cnetwork.primary.args.density = real_density;
if ~silent
    fprintf('-- Computing density\n'); aux.flushout();
    real_density % density = (number_of_links - loops) / max_number_of_links; where max_number_of_links = (Chi*(Chi-1) * l^2).
    theoretical_average_density = 1 - (1 - c * (c-1) / (Chi * (Chi - 1) * l^2) )^m
    total_number_of_messages_really_stored = log2(1 - real_density) / log2(1 - c*(c-1) / (Chi*(Chi-1)*l.^2))
    number_of_nodes = n % total number of nodes (l * c)
    number_of_active_links = (nnz(cnetwork.primary.net) - nnz(diag(cnetwork.primary.net))) / 2 % total number of active links
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
