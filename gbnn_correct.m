function partial_messages = gbnn_correct(varargin)
%
% partial_messages = gbnn_correct(cnetwork, partial_messages, ...
%                                                                                  l, c, Chi, ...
%                                                                                  iterations, ...
%                                                                                  k, guiding_mask, gamma_memory, threshold, propagation_rule, filtering_rule, tampering_type, ...
%                                                                                  residual_memory, concurrent_cliques, GWTA_first_iteration, GWTA_last_iteration, ...
%                                                                                  silent)
%
% Feed a network and partially tampered messages, and will let the network try to 'remember' a message that corresponds to the given input. The function will return the recovered message(s) (but no error rate, see gbnn_test.m for this purpose).
%
% This function supports named arguments, use it like this:
% gbnn_correct('cnetwork', mynetwork, 'partial_messages', thriftymessagestest, 'l', 4, 'c', 3)
% 


% == Importing some useful functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% == Arguments processing
% List of possible arguments and their default values
arguments_defaults = struct( ...
    ... % Mandatory
    'cnetwork', [], ...
    'partial_messages', [], ...
    'l', 0, ...
    'c', 0, ...
    ...
    ... % 2014 sparse enhancement
    'Chi', 0, ...
    ...
    ... % Test settings
    'iterations', 1, ...
    ...
    ... % Tests tweakings and rules
    'guiding_mask', [], ...
    'threshold', 0, ...
    'gamma_memory', 0, ...
    'residual_memory', 0, ...
    'propagation_rule', 'sum', ...
    'filtering_rule', 'wta', ...
    'tampering_type', 'erase', ...
    'concurrent_cliques', 1, ... % 1 is disabled, > 1 enables and specify the number of concurrent messages/cliques to decode concurrently
    'GWTA_first_iteration', false, ...
    'GWTA_last_iteration', false, ...
    'k', 1, ...
    ...
    ... % Debug stuffs
    'silent', false);

% Process the arguments
arguments = aux.getnargs(varargin, arguments_defaults, true);

% Load variables into local namespace (called workspace in MatLab)
aux.varspull(arguments);

% == Sanity Checks
if isempty(cnetwork) || isempty(partial_messages) || l == 0 || c == 0
    error('Missing arguments: cnetwork, partial_messages, l and c are mandatory!');
end

variable_length = false;
if isvector(c) && ~isscalar(c)
    variable_length = true;
end

if ~ischar(propagation_rule)
    if iscell(propagation_rule); error('propagation_rule is a cell, it should be a string! Maybe you did a typo?'); end;
    propagation_rule = 'sum';
end
if ~ischar(filtering_rule)
    if iscell(filtering_rule); error('filtering_rule is a cell, it should be a string! Maybe you did a typo?'); end;
    filtering_rule = 'wta';
end
if ~ischar(tampering_type)
    if iscell(tampering_type); error('tampering_type is a cell, it should be a string! Maybe you did a typo?'); end;
    tampering_type = 'erase';
end

% == Init data structures and other vars - DO NOT TOUCH
sparse_cliques = true; % enable the creation of sparse cliques if Chi > c (cliques that don't use all available clusters but just c clusters per one message)
if Chi <= c
    Chi = c; % Chi can't be < c, thus here we ensure that
    sparse_cliques = false;
end
n = Chi * l; % total number of nodes ( = length of a message = total number of characters slots per message)

% Smart messages management: if user provide a non thrifty messages matrix, we convert it on-the-fly (a lot easier for users to manage in their applications since they can use the same messages to learn and to test the network)
if ~islogical(partial_messages) && any(partial_messages(:) > 1)
    partial_messages = gbnn_messages2thrifty(partial_messages, l);
end

mpartial = size(partial_messages, 2); % mpartial = number of messages to reconstruct. This is also equal to tampered_messages_per_test in gbnn_test.m

% -- A few error checks
if numel(c) ~= 1 && numel(c) ~= 2
    error('c contains too many values! numel(c) should be equal to 1 or 2.');
end
if n ~= size(partial_messages, 1)
    error('Provided arguments Chi and L do not match with the size of partial_messages.');
end
if concurrent_cliques > 1 && k > n && ~(strcmpi(filtering_rule, 'WTA') || strcmpi(filtering_rule, 'LKO') || strcmpi(filtering_rule, 'GWTA') || strcmpi(filtering_rule, 'GLKO'))
    error('k cannot be > Chi*L, this means that you are trying to use too many concurrent_cliques for a too small network! Try to lower concurrent_cliques or to increase the size of your network.');
end


% #### Correction phase
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
            propag = cnetwork * partial_messages; % Propagate the previous message state by using a simple matrix product (equivalent to similarity/likelihood? or is it more like a markov chain convergence?). Eg: if network = [0 1 0 ; 1 0 0], partial_messages = [1 1 0] will match quite well: [1 1 0] while partial_messages = [0 0 1] will not match at all: [0 0 0]
        else % MatLab cannot do matrix multiplication on logical matrices...
            propag = double(cnetwork) * double(partial_messages);
        end
    % Else error, the propagation_rule does not exist
    else
        error('Unrecognized propagation_rule: %s', propagation_rule);
    % TODO: sum-of-max use spfunc() or arrayfun() to do a custom matrix computation per column: first an and, then a reshape to get only nodes per cluster on one colum, then any, then we have our new message! Or compute sos then diff(sos - som) then sos - diff(sos-sum)
    % TODO: Sum-of-Max = for each node, compute the sum of incoming link per cluster (thus if a node is connected to 4 other nodes, but 3 of them belong to the same cluster, the score will be 2 = 1 for the node alone in its cluster + 1 for the other 3 belonging to the same cluster).
    % TODO: Normalized Sum-of-Sum = for each node, compute the sum of sum but divide the weight=1 of each incoming link by the number of activated node in the cluster where the link points to (thus if a node is connected to 4 other nodes, but 3 of them belong to the same cluster, the score will be 1 + 1/3 + 1/3 + 1/3 = 2). The score will be different than Sum of Max if concurrent_cliques is enabled (because then the number of activated nodes can be higher and divide more the incoming scores).
    end
    if gamma_memory > 0
        propag = propag + (gamma_memory .* partial_messages); % memory effect: keep a bit of the previous nodes scores
    end
    if threshold > 0 % activation threshold: set to 0 all nodes with a score lower than the activation threshold (ie: those nodes won't emit a spike)
        propag(propag < threshold) = 0;
    end;

    % -- Semi-vectorized version 1 - fast but very memory consuming
    % Using the adjacency matrix as an adjacency list of neighbors per node: we just fetch the neighbors indices and then compute the final message
%        propag_idx = mod(find(partial_messages)-1, c*l)+1; % get the indices of the currently activated nodes (we use mod because we compute at once the indices for all messages, and the indices will be off starting from the second message)
%        activated_neighbors = cnetwork(:, propag_idx); % propagation: find the list of neighbors for each currently activated node = just use the current nodes indices in the adjacency list. Thus we get one list of activated neighbors per currently activated node.
%        messages_idx = [0 cumsum(sum(partial_messages))]; % now we need to unstack all lists of neighbors per message instead of per node (we will get one list of neighbor per activated neuron in one message, thus we need to unstack the matrix into submatrixes: one per message)
%        propag = sparse(c*l, mpartial); % preallocate the propag matrix
%        parfor i=1:mpartial % now we will sum all neighbors activations into one message (because we have many submatrices = messages containing lists of activated neighbors per currently activated node, but we want one list of activated neighbors for all nodes = one message)
%            propag(:,i) = sum( activated_neighbors(:,messages_idx(i)+1:messages_idx(i+1)) , 2); % TODO: Vectorize this! find a way to do it without a for loop!
%        end

    % -- Semi-vectorized version 2 - slower but easy on memory
    % same as version 1 but to lower memory footprint we compute the activated_neighbors in the loop instead of outside.
%        propag_idx = mod(find(partial_messages)-1, c*l)+1;
%        cluster_idx = [0 cumsum(sum(partial_messages))];
%        propag = sparse(c*l, mpartial);
%        parfor i=1:mpartial
%            propagidx_per_cluster = propag_idx(cluster_idx(i)+1:cluster_idx(i+1));
%            activated_neighbors = cnetwork(:, propagidx_per_cluster);
%            propag(:,i) = sum( activated_neighbors , 2);
%        end


    % 2- Filtering rules aka activation rules (apply a rule to filter out useless/interfering nodes)

    % -- Vectorized version - fastest!
    out = logical(sparse(size(partial_messages,1), size(partial_messages,2))); % empty binary sparse matrix, it will later store the next network state after winner-takes-all is applied
    % Winner-take-all : per cluster, keep only the maximum score node active (if multiple nodes share the max score, we keep them all activated). Thus the WTA is based on score value, contrarywise to k-WTA which is based on the number of active node k.
    if strcmpi(filtering_rule, 'wta')
        % The idea is that we will break the clusters and stack them along as a single long cluster spanning several messages, so that we can do a WTA in one pass (with a single max), and then we will unstack them back to their correct places in the messages
        propag = reshape(propag, l, mpartial * Chi); % reshape so that we can do the WTA by a simple column-wise WTA (and it's efficient in MatLab since matrices - and even more with sparse matrices - are stored as column vectors, thus it efficiently use the memory cache since this is the most limiting factor above CPU power). See also: Locality of reference.
        winner_value = max(propag); % what is the maximum output value (considering all the nodes in this character)
        if ~aux.isOctave()
            out = logical(bsxfun(@eq, propag, winner_value)); % Get the winners by filtering out nodes that aren't equal to the max score. Use bsxfun(@eq) instead of ismember.
        else
            out = logical(sparse(bsxfun(@eq, propag, winner_value))); % Octave does not overload bsxfun with sparse by default, so bsxfun removes the sparsity, thus the propagation at iteration > 1 will be horribly slow! Thus we have to put it back manually. MatLab DOES overload bsxfun with sparsity so this only applies to Octave. See: http://octave.1599824.n4.nabble.com/bsxfun-and-sparse-matrices-in-Matlab-td3867746.html
        end
        out = and(propag, out); % IMPORTANT NO FALSE WINNER TRICK: if sparse_cliques or variable_length, we deactivate winning nodes that have 0 score (but were chosen because there's no other activated node in this cluster, and max will always return the id of one node! Thus we have to compare with the original propagation matrix and see if the node was really activated before being selected as a winner).
        out = reshape(out, n, mpartial);
    % k-Winner-take-all : keep the best first k nodes having the maximum score, over the whole message. This is kind of a cheat because we must know the original length of the messages, the variable k is a way to tell the algorithm some information we have to help convergence
    elseif strcmpi(filtering_rule, 'kwta')
        propag = reshape(propag, l, mpartial * Chi); % reshape so that we can do the WTA by a simple column-wise WTA (and it's efficient in MatLab since matrices - and even more with sparse matrices - are stored as column vectors, thus it efficiently use the memory cache since this is the most limiting factor above CPU power). See also: Locality of reference.
        [~, idxs] = sort(propag, 'descend');
        idxs = bsxfun( @plus, idxs, 0:l:((mpartial * n)-1) );
        [I, J] = ind2sub(size(propag), idxs(1:k, :));
        out = logical(sparse(I, J, 1, l, mpartial * Chi));
        out = and(propag, out); % IMPORTANT NO FALSE WINNER TRICK: if sparse_cliques or variable_length, filter out winning nodes with 0 score (selected because there's no other node with any score in this cluster, see above in filtering_rule = 0)
        out = reshape(out, n, mpartial);
    % One Global Winner-take-all: only keep one value, but at the last iteration keep them all
    elseif strcmpi(filtering_rule, 'ogwta') && iter < iterations
        [~, winner_idxs] = max(propag); % get indices of the max score for each message
        winner_idxs = bsxfun(@plus, winner_idxs,  0:n:n*(mpartial-1)); % indices returned by max are per-column, here we add the column size to offset the indices, so that we get true MatLab indices
        winner_idxs = winner_idxs(propag(winner_idxs) > 0); % No false winner trick: we check the values returned by the k best indices in propag, and keep only indices which are linked to a non null value.
        if nnz(winner_idxs) > 0 % If there is at least one non null value (there's at least one remaining index after the no false winner trick)
            [I, J] = ind2sub(size(propag), winner_idxs); % extract the sparse matrix non null values indices
            out = logical(sparse(I, J, 1, n, mpartial)); % create the logical sparse matrix
        else % Else there's not even one non null value, this means that all messages have 0 max score (the network shut down), then we just have to create an empty output
            out = logical(sparse(n, mpartial)); % create an empty logical sparse matrix
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
        idxs = bsxfun(@plus, idxs,  0:n:n*(mpartial-1)); % indices returned by sort are per-column, here we add the column size to offset the indices, so that we get true MatLab indices
        winner_idxs = idxs(propag(idxs) > 0); % No false winner trick: we check the values returned by the k best indices in propag, and keep only indices which are linked to a non null value.
        [I, J] = ind2sub(size(propag), winner_idxs); % extract the necessary I and J (row and columns) indices vectors to create a sparse matrix
        out = logical(sparse(I, J, 1, n, mpartial)); % finally create a logical sparse matrix
    % WinnerS-take-all: per cluster, select the kth best score, and accept all nodes with a score greater or equal to this score - similar to kWTA but all nodes with the kth score are accepted, not only a fixed number k of nodes
    % Global WinnerS-take-all: same as WsTA but per the whole message
    % Both are implemented the same way, the only difference is that for global we process winners per message, and with local we process them per cluster
    elseif strcmpi(filtering_rule, 'WsTA') || strcmpi(filtering_rule, 'GWsTA')
        % Local WinnerS-take-all: Reshape to get only one cluster per column (instead of one message per column)
        if strcmpi(filtering_rule, 'WsTA')
            propag = reshape(propag, l, mpartial * Chi);
        end

        max_scores = sort(propag,'descend');
        kmax_score = max_scores(k, :);
        kmax_score(kmax_score == 0) = realmin(); % No false winner trick: better version of the no false winner trick because we replace 0 winner scores by the minimum real value, thus after we will still get a sparse matrix (else if we do the trick only after the bsxfun, we will get a matrix filled by 1 where there are 0 and the winner score is 0, which is a lot of memory used for nothing).
        out = logical(bsxfun(@ge, propag, kmax_score));
        if aux.isOctave(); out = sparse(out); end; % Octave's bsxfun breaks the sparsity...
        out = and(propag, out); % No false winner trick: avoids that if the kth max score is in fact 0, we choose 0 as activating score (0 scoring nodes will be activated, which is completely wrong!). Here we check against the original propagation matrix: if the node wasn't activated then, it should be now.

        % Local WinnerS-take-all: Reshape back the clusters into messages
        if strcmpi(filtering_rule, 'WsTA')
            out = reshape(out, n, mpartial);
        end

    % Loser-kicked-out (locally, per cluster, we kick loser nodes with min score except if min == max of this cluster).
    % Global Loser-Kicked-Out (deactivate all nodes with min score in the message)
    % Both are implemented the same way, the only difference is that for global we process losers per message, and with local we process them per cluster
    elseif strcmpi(filtering_rule, 'LKO') || strcmpi(filtering_rule, 'GLKO')
        % Local Losers-Kicked-Out: Reshape to get only one cluster per column (instead of one message per column)
        if strcmpi(filtering_rule, 'LKO')
            propag = reshape(propag, l, mpartial * Chi);
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
            out = reshape(out, n, mpartial);
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
            propag = reshape(propag, l, mpartial * Chi);
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
            out = reshape(out, n, mpartial);
        end
    % Concurrent Global k-Winners-take-all
    % CkGWTA = kGWTA + trimming out all scores that are below the k-th max score (so that if there are shared nodes between multiple messages, we will get less than c active nodes at the end)
    elseif strcmpi(filtering_rule, 'CGkWTA')
        % 1- kGWTA
        [~, idxs] = sort(propag, 'descend'); % Sort scores with best scores first
        idxs = idxs(1:k,:); % extract the k best scores indices (for propag)
        idxs = bsxfun(@plus, idxs,  0:n:n*(mpartial-1)); % indices returned by sort are per-column, here we add the column size to offset the indices, so that we get true MatLab indices
        winner_idxs = idxs(propag(idxs) > 0); % No false winner trick: we check the values returned by the k best indices in propag, and keep only indices which are linked to a non null value.
        [I, J] = ind2sub(size(propag), winner_idxs); % extract the necessary I and J (row and columns) indices vectors to create a sparse matrix
        temp = propag(idxs); % MatLab compatibility...
        propag2 = sparse(I, J, temp(temp > 0), n, mpartial); % finally create a sparse matrix, but contrarywise to the original implementation of kGWTA, we keep the full scores here, we don't turn it yet into binary, so that we can continue processing in the second stage.

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

    % Guiding mask
    if ~isempty(guiding_mask) % Apply guiding mask to filter out useless clusters. TODO: the filtering should be directly inside the propagation rule at the moment of the matrix product to avoid useless computations (but this is difficult to do this in MatLab in a vectorized way...).
        out = reshape(out, l, mpartial * Chi);
        if ~aux.isOctave()
            out = bsxfun(@and, out, guiding_mask);
        else
            out = logical(sparse(bsxfun(@and, out, guiding_mask)));
        end
        out = reshape(out, n, mpartial);
    end

    % Residual memory
    if residual_memory > 0
        out = out + (residual_memory .* partial_messages); % residual memory: previously activated nodes lingers a bit (a fraction of their activation score persists) and participate in the next iteration
    end

    partial_messages = out; % set next messages state as current
    if ~silent; aux.printtime(toc()); end;
end

% -- After-convergence post-processing
if residual_memory > 0 % if residual memory is enabled, we need to make sure that values are binary at the end, not just near-binary (eg: values of nodes with 0.000001 instead of just 0 due to the memory addition), else the error can't be computed correctly since some activated nodes will still linger after the last WTA!
    partial_messages = max(round(partial_messages), 0);
end

% At the end, partial_messages contains the recovered, corrected messages (which may still contain errors, but the goal is to have a perfect recovery!)
% The performance, error test must be done by yourself, by comparing the returned partial_messages with your original messages.

end % endfunction
