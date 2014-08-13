function [error_rate, theoretical_error_rate, test_stats, error_per_message, testset] = gbnn_test(varargin)
%
% [error_rate, theoretical_error_rate, test_stats] = gbnn_test(cnetwork, thriftymessagestest, ...
%                                                                                  erasures, iterations, tampered_messages_per_test, tests, ...
%                                                                                  enable_guiding, gamma_memory, threshold, propagation_rule, filtering_rule, tampering_type, ...
%                                                                                  residual_memory, concurrent_cliques, no_concurrent_overlap, concurrent_successive, GWTA_first_iteration, GWTA_last_iteration, ...
%                                                                                  silent)
%
% Feed a network and a matrix of thrifty messages from which to pick samples for test, and this function will automatically sample some messages, tamper them, and then try to correct them. Finally, the error rate over all the processed messages will be returned.
%
% This function supports named arguments, use it like this:
% gbnn_test('cnetwork', mynetwork, 'thriftymessagestest', thriftymessagestest, 'l', 4, 'c', 3)
%
% -- Test variables
%- cnetwork : specify the network to use that you previously learned. This is a struct containing the network and which should also contain the parameters (eg: l, c and Chi).
%- thriftymessagestest : matrix of messages that will be used as a test set. NOTE: you can either provide a thrifty messages matrix (only composed of 0's and 1's, eg: 001010) or a full messages matrix (eg: 23040), so that you can use the same format of messages matrix both in gbnn_learn.m and gbnn_test.m
%- erasures : number of characters that will be erased (or noised if tampering_type == "noise") from a message for test. NOTE: must be lower than c!
%- iterations : number of iterations to let the network converge
%- tampered_messages_per_test : number of tampered messages to generate and try per test (for maximum speed, set this to the maximum allowed by your memory and decrease the number tests)
%- tests : number of tests (number of batch of tampered messages to test)
% NOTE2: at test phase, iteration 1 will always be very very fast, and subsequent iterations (but mostly the second one) will be very slow in comparison. This is normal because after the first propagation, the number of activated nodes will be hugely bigger (because one node is connected to many others), thus producing this slowing effect.
% NOTE3: setting a high erasures number will slow down the test phase after first iteration, because it will exacerbate the number of potential candidates (ie: the maximum score will be lower since there is less activated nodes at the first propagation iteration, and thus many candidate nodes will attain the max score, and thus will be selected by WTA). This is an extension of the effect described in note2.
%
% -- 2014 update ("Storing Sparse Messages in Networks of Neural Cliques" by Behrooz, Berrou, Gripon, Jiang)
%- guiding_mask : guide the decoding by focusing only on the clusters where we know the characters may be (by cheating and using the initial message as a guide, but a less efficient guiding mask could be generated by other means). This is only useful if you enable sparse_cliques (Chi > c).
%- gamma_memory : memory effect: keep a bit of the previous nodes value. This helps when the original message is sure to contain the correct bits (ie: works with messages partially erased, but maybe not with noise)
%- threshold : activation threshold. Nodes having a score below this threshold will be deactivated (ie: the nodes won't produce a spike). Unlike the article, this here is just a scalar (a global threshold for all nodes), rather than a vector containing per-cluster thresholds.
%- propagation_rule : also called "dynamic rule", defines the type of propagation algorithm to use (how nodes scores will be computed at next iteration). "sum"=Sum-of-Sum ; "normalized"=Normalized-Sum-of-Sum ; "max"=Sum-of-Max % TODO: only 0 SoS is implemented right now!
%- filtering_rule : also called "activation rule", defines the type of filtering algorithm (how to select the nodes that should remain active), generally a Winner-take-all algorithm in one of the following (case insensitive): 'WTA'=Winner-take-all (keep per-cluster nodes with max activation score) ; 'kWTA'=k-Winners-take-all (keep exactly k best nodes per cluster) ; 'oGWTA'=One-Global-Winner-take-all (same as GWTA but only one node per message can stay activated, not all nodes having the max score) ; 'GWTA'=Global Winner-take-all (keep nodes that have the max score in the whole network) ; 'GkWTA'=Global k-Winner-take-all (keep nodes that have a score equal to one of k best scores) ; 'WsTA'=WinnerS-take-all (per cluster, select the kth best score, and accept all nodes with a score greater or equal to this score - similar to kWTA but all nodes with the kth score are accepted, not only a fixed number k of nodes) ; 'GWsTA'=Global WinnerS-take-all (same as WsTA but per the whole message) ; 'GLKO'=Global Loser-kicked-out (kick nodes that have the worst score globally) ; 'GkLKO'=Global k-Losers-kicked-out (kick all nodes that have a score equal to one of the k worst scores globally) ; 'LKO'=Losers-Kicked-Out (locally, kick k nodes with worst score per-cluster) ; 'kLKO'=k-Losers-kicked-out (kick k nodes with worst score per-cluster) ; 'CGkWTA'=Concurrent Global k-Winners-Take-All (kGWTA + trimming out all scores that are below the k-th max score) ; TODO : 10=Optimal-Global-Loser-Kicked-Out (GkLKO but with k = 1 to kick only one node per iteration)
%- tampering_type : type of message tampering in the tests. "erase"=erasures (random activated bits are switched off) ; "noise"=noise (random bits are flipped, so a 0 becomes 1 and a 1 becomes 0).
%
% -- Custom extensions
%- residual_memory : residual memory: previously activated nodes lingers a bit and participate in the next iteration.
%- concurrent_cliques : allow to decode multiple messages concurrently/simultaneously (can specify the number here).
%- no_concurrent_overlap : ensure that concurrent messages/cliques aren't overlapping, thus there won't be any shared fanal (with a big score).
%- concurrent_successive : instead of simultaneously stimulate the concurrent messages, stimulate them one after the other in succession. This is a bit like tagging/coloring the cliques to better handle them. For example, guiding_mask is adapted at each step to allow a cumulation of one more message at each step (ie: at first step there will be c clusters allowed in the first guiding_mask, then 2*c in the second step, then 3*c in the third, etc.) instead of allowing all correct clusters for all concurrent messages at once..
%- GWTA_first_iteration and GWTA_last_iteration : always use GWTA as the first/last filtering rule. Particularly useful with *G*LKO family of algorithms which need the first iteration to be GWTA to kickstart the process.
%- silent : silence all outputs
%


% == Importing some useful functions
aux = gbnn_aux; % works with both MatLab and Octave

% == Arguments processing
% List of possible arguments and their default values
arguments_defaults = struct( ...
    ... % Mandatory
    'cnetwork', [], ...
    'thriftymessagestest', [], ...
    ...
    ... % Test settings
    'erasures', 1, ...
    'iterations', 1, ...
    'tampered_messages_per_test', 1, ...
    'tests', 1, ...
    ...
    ... % Tests tweakings and rules
    'enable_guiding', false, ...
    'threshold', 0, ...
    'gamma_memory', 0, ...
    'residual_memory', 0, ...
    'propagation_rule', 'sum', ...
    'filtering_rule', 'wta', ...
    'tampering_type', 'erase', ...
    'GWTA_first_iteration', false, ...
    'GWTA_last_iteration', false, ...
    'enable_dropconnect', false, ...
    'dropconnect_p', 0.5, ...
    ...
    ... % Concurrency extension (simultaneous messages/cliques)
    'concurrent_cliques', 1, ... % 1 is disabled, > 1 enables and specify the number of concurrent messages/cliques to decode concurrently
    'no_concurrent_overlap', false, ...
    'concurrent_successive', false, ...
    'concurrent_disequilibrium', false, ...
    ...
    ... % Debug stuffs
    'silent', false);

% Process the arguments
arguments = aux.getnargs(varargin, arguments_defaults, true);

% Load variables into local namespace (called workspace in MatLab)
aux.varspull(arguments);
aux.varspull(cnetwork.primary.args);

% == Sanity Checks
if isempty(cnetwork) || isempty(thriftymessagestest) || isfield(cnetwork.primary.args, 'l') == 0 || isfield(cnetwork.primary.args, 'c') == 0
    error('Missing arguments: cnetwork, thriftymessagestest, cnetwork.primary.args.l and cnetwork.primary.args.c are mandatory!');
end

if ~exist('propagation_rule', 'var') || ~ischar(propagation_rule)
    if iscell(propagation_rule); error('propagation_rule is a cell, it should be a string! Maybe you did a typo?'); end;
    propagation_rule = 'sum';
end
if ~exist('filtering_rule', 'var') || ~ischar(filtering_rule)
    if iscell(filtering_rule); error('filtering_rule is a cell, it should be a string! Maybe you did a typo?'); end;
    filtering_rule = 'wta';
end
if ~exist('tampering_type', 'var') || ~ischar(tampering_type)
    if iscell(tampering_type); error('tampering_type is a cell, it should be a string! Maybe you did a typo?'); end;
    tampering_type = 'erase';
end

% == Show vars (just for the record, user can debug or track experiments using diary)
if ~silent
    % -- Network variables
    l
    c
    Chi

    % -- Test variables
    alpha = erasures / c % percent of erased bits per message
    erasures
    iterations
    tampered_messages_per_test
    tests

    % -- 2014 update
    gamma_memory
    threshold
    enable_guiding
    propagation_rule
    filtering_rule
    tampering_type

    % -- Custom extensions
    residual_memory
    concurrent_cliques
    no_concurrent_overlap
    concurrent_successive
end



% == Init data structures and other vars - DO NOT TOUCH

% Smart messages management: if user provide a non thrifty messages matrix, we convert it on-the-fly (a lot easier for users to manage in their applications since they can use the same messages to learn and to test the network)
if ~islogical(thriftymessagestest) && any(thriftymessagestest(:) > 1)
    thriftymessagestest = gbnn_messages2thrifty(thriftymessagestest, l);
end

% Get the count of test messages
mtest = size(thriftymessagestest, 1);

% -- A few error checks
if erasures > c
    error('Erasures > c which is not possible.');
end
if numel(c) ~= 1 && numel(c) ~= 2
    error('c contains too many values! numel(c) should be equal to 1 or 2.');
end
if n ~= size(thriftymessagestest, 2)
    error('Provided arguments Chi and L do not match with the size of thriftymessagestest.');
end

if nargin > 3
    error_per_message = logical(sparse(1, tests*tampered_messages_per_test));
    testset = logical(sparse(n, tests*tampered_messages_per_test));
end

if ~silent; totalperf = cputime(); end; % for total time perfs

% #### Test phase
% == Run the test (error correction) and compute error rate (in reconstruction of original message without erasures)
if ~silent; fprintf('#### Testing phase (error correction of tampered messages with %i erasures)\n', erasures); aux.flushout(); end;
%thriftymessagestest = thriftymessages; % by default, use to test the same set as for learning
err = 0; % error score
derr = 0; % euclidian error distance
similarity_measure = 0; % similarity between the corrected messages and the initial messages (number of matching characters over the total number of the biggest message)
matching_measure = 0; % does the corrected message contains at least all the fanals of the initial message?
for t=1:tests % TODO: replace by parfor (regression from past versions to allow for better compatibility because Octave cannot do parallel processing, but parfor should work with little modifications)
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

    % If no_concurrent_overlap is enabled, we will regenerate messages that overlap, so that no shared fanal exist in the final set of messages
    no_concurrent_overlap_flag = false;
    overlap_idxs = [];
    mtogen = tampered_messages_per_test;
    while (~no_concurrent_overlap_flag)

        % Generate random indices to randomly choose messages
        % At first iteration, we generate the whole set of messages. Then subsequent iterations only serve (when concurrent_cliques > 1 and no_concurrent_overlap is true) to generate replacement messages (messages that will replace the previously overlapping messages).
        %rndidx = unidrnd(mtest, [mconcat tampered_messages_per_test]); % TRICKS: unidrnd(m, [SZ]) is twice as fast as unidrnd(m, dim1, dim2)
        rndidx = randi([1 mtest], mconcat, mtogen); % mtest is the total number of messages in the test set (available to be picked up), mconcat is the number of concurrent messages that we will squash together, tampered_messages_per_test is the number of messages we will try to correct per test (number of messages to test per batch).

        % Fetch the random messages from the generated indices
        inputm = thriftymessagestest(rndidx,:)'; % just fetch the messages and transpose them so that we have one sparsemessage per column (we don't generate them this way even if it's possible because of optimization: any, or, and sum are more efficient column-wise than row-wise, as any other MatLab/Octave function).
        % Add or replace messages?
        if ~no_concurrent_overlap || isempty(overlap_idxs) % No overlap, just save the messages
            init = inputm; % backup the original message before tampering, we will use the original to check if the network correctly corrected the erasure(s)
        else % Else, we had overlapping messages in the previous while iteration, now we replace the overlapping messages by the new ones (so that we won't move around the previously generated messages, we are thus guaranteed that we won't produce more overlapping messages at replacement, we can only get better)
            init(:, overlap_idxs(:)) = inputm; % In-place replacement of overlapping messages by other randomly choosen messages.
            overlap_idxs = []; % empty the overlapping indices, so that we won't replace the same indices by mistake at next iteration
        end
        %if ~debug; clear rndidx; end; % clear up memory - DEPRECATED because it violates the transparency (preventing the parfor loop to work)

        % Overlapping detection and correction
        no_concurrent_overlap_flag = true; % if there's no concurrent_cliques or no_concurrent_overlap is false or if the overlap was fixed, the flag is enabled so that the while loop can stop.
        if concurrent_cliques > 1 && (~silent || no_concurrent_overlap) % else, in concurrent case, we must check if there is any overlap (and compute the concurrent_overlap_rate just for info)
            % Mix up init (untampered messages) but keep the number of sharing
            init_overlaps = reshape(init, n*tampered_messages_per_test, concurrent_cliques)'; % stack concurrent messages (the ones that will be merged together) side-by-side
            init_overlaps = sum(init_overlaps) > 1; % sum (instead of any) to get all shared fanals: they will have a score > 1
            concurrent_overlap_rate = nnz(init_overlaps) / (nnz(init)/concurrent_cliques); % concurrent overlap rate is the real frequency of having an overlap, which we compute as the number of overlapped characters divided by the mean number of characters per messages (here the division by the number of messages is implicit).

            % If there is any overlap and no_concurrent_overlap is enabled, detect which messages are overlapping
            if no_concurrent_overlap && concurrent_overlap_rate > 0
                no_concurrent_overlap_flag = false; % we need another while iteration

                % Detect the indices of overlapping messages
                overlap_idxs = unique(ceil(find(init_overlaps) ./ n)); % detect the message index of overlaps in merged messages (this gives us one index per package of concurrent messages, but we can then deduce the missing indices)
                mtogen = numel(overlap_idxs); % remember the number of messages we will have to generate again to replace the overlapping ones
                overlap_idxs = repmat(overlap_idxs, [concurrent_cliques, 1]); % expand indices to account for the unmerged messages (multiply by concurrent_cliques)
                offsets = (1:tampered_messages_per_test:tampered_messages_per_test * concurrent_cliques) - 1; % offset to align indices to unmerged messages (the first row is aligned, but all the others must be aligned to each concurrent message)
                overlap_idxs = bsxfun(@plus, overlap_idxs, offsets'); % apply offset
                %init(:, overlap_idxs) = []; % DEPRECATED: in-place remove, but then we can only append newly generated messages but they will unalign the other messages which won't be merged together like before but with other messages, and thus we may get even more overlapping!
            end
        end
    end
    inputm = init; % Finally, set inputm with init: we will work on inputm but leave init as a backup to later check the error correction performances

    % Show concurrent_overlap_rate
    if concurrent_cliques > 1 && ~silent
        concurrent_overlap_rate
    end

    % 2- randomly tamper them (erasure or noisy bit-flipping of a few characters)
    % -- Random erasure of random active characters (which the network is more tolerant than noise, which is normal and described in modern error correction theory)
    if strcmpi(tampering_type, 'erase')
        % The idea is that we will randomly pick several characters to erase per message (by extracting all nonzeros indices, order per-column/message, and then shuffling them to finally select only a few indices per column to point to the characters we will erase)
        [~, idxs] = sort(inputm, 'descend'); % sort the messages to get the indices of the nonzeros values, but still organized per-column (which find() doesn't provide)
        idxs = idxs(1:c, :); % memory efficiency: trim out indices of the zero values (since we are sure that at most a message contains c characters)
        idxs = aux.shake(idxs); % per-column shuffle indices! This is how we randomly pick characters.
        idxs = idxs(1:erasures, :); % select the number of erasures we want
        idxs = bsxfun(@plus, idxs, 0:n:n*(tampered_messages_per_test*concurrent_cliques-1) ); % offset indices to take account of the column (since sort resets indices count per column)
        idxs(idxs == 0) = []; % remove non valid indices (if variable_length, some characters may have less than the number of characters we want to erase) TODO: ensure that a variable_length message keeps at least 2 nodes
        inputm(idxs(1:erasures, :)) = 0; % erase those characters
    % -- Random noise (bit-flipping of randomly selected characters)
    elseif strcmpi(tampering_type, 'noise')
        % The idea is simple: we generate random indices to be "noised" and we bit-flip them using modulo.
        %idxs = unidrnd(n, [erasures mconcat*tampered_messages_per_test]); % generate random indices to be tampered
        idxs = randi([1 n], erasures, mconcat*tampered_messages_per_test); % generate random indices to be tampered
        idxs = bsxfun(@plus, idxs, 0:n:n*(tampered_messages_per_test*concurrent_cliques-1) ); % offset indices to take account of the column = message (since sort resets indices count per column)
        inputm(idxs) = mod(inputm(idxs) + 1, 2); % bit-flipping! simply add one to all those entries and modulo one, this will effectively bit-flip them.
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
%            inputm(charstart:charend, j+1:j+tampered_messages_per_test) = 0; % for each character (in a random order), we tamper it (precisely we tamper the thrifty code, hence why we copy a vector of zeros)
%        end
%    end

    % If concurrent_cliques is enabled, we must mix up the messages together after we've done the characters erasure
    if concurrent_cliques > 1
        % Mix up init (untampered messages)
        init_mixed = any( reshape(init, n*tampered_messages_per_test, concurrent_cliques)' ); % mix up messages (by stacking concurrent_cliques messages side-by-side and then summing/anying them)
        %init = reshape(init, tampered_messages_per_test, n)'; % WRONG % unstack the messages vector into a matrix with one mixed sparsemessage per column
        init_mixed = reshape(init_mixed', n, tampered_messages_per_test); % unstack the messages vector into a matrix with one mixed sparsemessage per column

        % Only if we don't compute in succession, we mix up the tampered messages now (else we will do that at the end after convergence)
        if ~concurrent_successive
            init = init_mixed; % remove unneeded unmixed init
            % Mix up the tampered messages
            inputm = reshape(inputm, n*tampered_messages_per_test, concurrent_cliques)';
            inputm = any(inputm);
            inputm = reshape(inputm', n, tampered_messages_per_test);
        end
    end

    if ~silent; aux.printtime(toc()); end;

    % -- Prediction step: feed the tampered message to the network and wait for it to converge to a stable state, hopefully the corrected message.
    %if ~silent; fprintf('-- Feed to network and wait for convergence\n'); aux.flushout(); end;
    if ~silent; tperf = cputime(); end;

    % Guiding mask preparation
    % if enabled, prepare the guiding mask (the list of clusters that we will keep, all the other nodes from other clusters will be set to 0). This guiding mask can be defined manually if you want, here to do it automatically we compute it from the initial untampered messages, thus we will keep nodes activated only in the clusters where there were activated nodes in the initial message.
    guiding_mask = sparse([]);
    if enable_guiding

        % Normal case (no concurrent_cliques or not concurrent_successive): we simply stack fanals of the same cluster together and merge them to get the mask (if at least one fanal in this cluster was enabled, the mask bit will be 1)
        % Thus we get one long guiding mask which is a big vector containing many sub-vectors, each one corresponding to a message
        if ~(concurrent_successive && concurrent_cliques > 1)
            guiding_mask = sparse(1, Chi * tampered_messages_per_test); % pre-allocating
            guiding_mask = any(reshape(init, l, Chi * tampered_messages_per_test)); % reshape so that we get all fanals from one cluster per column, and stack all clusters/columns of a message side-by-side, and do the same for all the other messages. Then use any to compile all fanals per cluster into one, which is the mask for this cluster (0 if disabled, 1 if this cluster contains a correct fanal). At the end we get a binary vector, which is in fact the stacking of multiple vectors: one binary vector per message (this message stacking allows us to vectorize the generation and applying of the guiding mask). Note: any is better than sum in our case, and it's also faster and keeps the logical datatype!

        % Concurrent successive case: we want that at each step the guiding mask gets adapted: for the first step, only one message/clique will be recovered, thus the guiding mask must only contain the bits for this message (c bits set to 1 = c clusters will be allowed). Then for the second step, we will try to keep the previous messages + recover a second one, thus we want the guiding mask to now allow for 2*c clusters/bits. And on and on for each subsequent concurrent message.
        % Thus we will construct a matrix of guiding_masks (one guiding_mask/vector for each step), and we will construct each guiding_mask cumulatively (each subsequent guiding_mask will keep the bits of all previous guiding_masks)
        else
            guiding_mask = sparse(concurrent_cliques, Chi * tampered_messages_per_test); % pre-allocating
            gmask_tmp = sparse(1, Chi*tampered_messages_per_test); % this is where we will store the previous guiding mask, so that we can cumulatively append/allow new clusters/bits.
            % For each concurrent message, we will make an adapted guiding mask, hence the final guiding mask won't be a vector but a matrix (a mega vector of subvectors like before, but for each step the concurrent_successive processing)
            for cc=1:concurrent_cliques
                idxs = (tampered_messages_per_test*(cc-1)+1):(tampered_messages_per_test*cc); % get the indices of the messages for the current step
                gmask_tmp = or(any(reshape(init(:, idxs), l, Chi * tampered_messages_per_test)), gmask_tmp); % just like in the normal case, reshape to stack fanals of the same cluster together and then squash them using any, but in addition we append these bits to the previous step's guiding_mask.
                guiding_mask(cc,:) = gmask_tmp; % store the guiding_mask for this step
            end
        end

        % Make sure this is a logical (binary) datatype, else we will run into troubles with bsxfun(@and, ...)
        guiding_mask = logical(guiding_mask);
    end

    % Remove the unmixed init, we won't need it anymore as afterwards from here we will only use init to check the error rate of the final converged message, which is mixed
    if concurrent_cliques > 1
        init = init_mixed;
    end

    % Correction of the tampered messages
    if ~(concurrent_successive && concurrent_cliques > 1) % Normal case: just feed the messages (a matrix containing messages as vectors) and wait for convergence
        %inputm_full = inputm;
        % Correct and wait for convergence!
        [inputm, propag] = gbnn_correct('cnetwork', cnetwork, 'partial_messages', inputm, ...
                                  'iterations', iterations, ...
                                  'guiding_mask', guiding_mask, 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', propagation_rule, 'filtering_rule', filtering_rule, ...
                                  'residual_memory', residual_memory, 'concurrent_cliques', concurrent_cliques, 'GWTA_first_iteration', GWTA_first_iteration, 'GWTA_last_iteration', GWTA_last_iteration, ...
                                  'enable_dropconnect', enable_dropconnect, 'dropconnect_p', dropconnect_p, ...
                                  'concurrent_disequilibrium', concurrent_disequilibrium, ...
                                  'silent', silent);
        % DEBUG: show the original input and the corrected input interleaved by column. Thank's to Peter Yu http://www.peteryu.ca/tutorials/matlab/interleave_matrices
        %a = aux.interleave(inputm_full, inputm, 2); full([sum(a); a])
        %keyboard
    % Concurrent_successive case: we won't feed the mixed messages at once but instead we will try to converge for one message at a time, and then at each step we append another concurrent message and try again to converge, etc.
    else
        inputm_full = inputm; % Backup the unmixed messages
        for cc=1:concurrent_cliques % Feed one message at a time, but cumulatively (we append concurrent messages to the previous ones)
            if ~silent; fprintf('=> Converging for successive clique %i\n', cc); aux.flushout(); end;
            % Prepare the input messages: extract only one message and leave the other concurrent messages for later
            if cc == 1 % First step: we just take the first messages (first in the sense of all concurrent messages)
                inputm = inputm_full(:,1:tampered_messages_per_test);
            else % Subsequent steps: we cumulatively append the bits of other messages on the previous (already converged) messages
                % Mix up the next clique with the previous message
                idxs = (tampered_messages_per_test*(cc-1)+1):(tampered_messages_per_test*cc); % get the indices of the messages for the current step
                inputm = or(inputm, inputm_full(:, idxs)); % cumulatively merge messages of the current step on the messages of previous steps
            end

            % Re-setup correct values for k, since the number of concurrent_cliques will vary for each step
            k = c*cc;
            if strcmpi(filtering_rule, 'kWTA') || strcmpi(filtering_rule, 'kLKO') || strcmpi(filtering_rule, 'WsTA')
                k = cc;
            end

            % Load the guiding mask for this step, but only if guiding is enabled (else the indexing will fail with an error since the guiding_mask won't be constructed)
            gmask = sparse([]);
            if enable_guiding
                gmask = guiding_mask(cc, :);
            end

            % Correct and wait for convergence!
            [inputm, propag] = gbnn_correct('cnetwork', cnetwork, 'partial_messages', inputm, ...
                                  'iterations', iterations, ...
                                  'k', k, 'guiding_mask', gmask, 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', propagation_rule, 'filtering_rule', filtering_rule, ...
                                  'residual_memory', residual_memory, 'concurrent_cliques', cc, 'GWTA_first_iteration', GWTA_first_iteration, 'GWTA_last_iteration', GWTA_last_iteration, ...
                                  'enable_dropconnect', enable_dropconnect, 'dropconnect_p', dropconnect_p, ...
                                  'concurrent_disequilibrium', concurrent_disequilibrium, ...
                                  'silent', silent);

        end
    end

    if ~silent
        fprintf('-- Propagation done!\n'); aux.flushout();
        aux.printcputime(cputime() - tperf, 'Propagation total elapsed cpu time is %g seconds.\n'); aux.flushout();
    end

    % -- Test score: compare the corrected message by the network with the original untampered message and check whether it's the same or not (partial or non correction are scored the same: no score)
    % If tampered_messages_per_test > 1, then the score is incremented per each unrecovered message, not per the whole pack
    %if ~silent; fprintf('-- Converged! Now computing score\n'); aux.flushout(); end;
    if ~silent; tic(); end;
    if tampered_messages_per_test > 1
        %err = err + sum(min(sum((init ~= inputm), 1), 1)); % this is a LOT faster than isequal() !
        err = err + nnz(sum((init ~= inputm), 1)); % even faster!
        derr = derr + sum(sum(init ~= inputm)); % error distance = euclidian distance to the correct message = mean number of bits that are wrong per message / number of bits per message = esperance that a bit is wrongly flipped
        similarity_measure = similarity_measure + (sum(inputm .* init, 1) / max(sum(inputm, 1), sum(init, 1))); % 1.0 if both messages are equal and of same length, the score will lower towards 0 if either some characters in the corrected message are wrong, or either if the corrected message contains more characters than the initial.
        matching_measure = matching_measure + (sum(inputm .* init, 1) / sum(init, 1)); % 1.0 if the corrected message contains at least the initial message (but the corrected message can contain more characters).
    else
        %err = err + ~isequal(init,inputm);
        err = err + any(init ~= inputm); % remove the useless sum(min()) when we only have one message to compute the error from, this cuts the time by almost half
        derr = derr + sum(init ~= inputm);
    end
    if nargout > 3
        indstart = 1+(t-1)*tampered_messages_per_test;
        indend = t*tampered_messages_per_test;
        error_per_message(:, indstart:indend) = any(init ~= inputm);
        testset(:, indstart:indend) = init;
    end
    if ~silent; aux.printtime(toc()); end;
    %if nnz(sum((init ~= inputm), 1)) > 0; keyboard; end;

end

% Compute error rate
error_rate = err / (tests * tampered_messages_per_test); % NOTE: if you use concurrent_cliques > 1, error_rate is not a good measure, because you artificially increase the probability of having a wrong message by concurrent_cliques times (since you're not testing one but concurrent_cliques messages at the same time), and there's no way to correct this biased estimator since we can't know which clique caused which bit flip (eg: concurrent_cliques = 3 and there are 3 wrong bits: are they caused by the three messages, or by only 1 and the other two are in fact corrects?). You should rather try error_distance in this case.
% Compute density
real_density = full(  (nnz(cnetwork.primary.net) - nnz(diag(cnetwork.primary.net))) / (Chi*(Chi-1) * l^2)  );
%real_density = 0;
%if ~strcmpi(propagation_rule, 'overlays')
%    real_density = full(  (nnz(cnetwork.primary.net) - nnz(diag(cnetwork.primary.net))) / (Chi*(Chi-1) * l^2)  );
%else
%    alltags = unique(nonzeros(cnetwork.primary.net));
%    for tg = 1:numel(alltags)
%        real_density = real_density + full(  (nnz(cnetwork.primary.net == alltags(tg)) - nnz(diag(cnetwork.primary.net == alltags(tg)))) / (Chi*(Chi-1) * l^2)  );
%    end
%    real_density = real_density / numel(alltags);
%end

% Compute theoretical error rate
theoretical_error_rate = -1;
if strcmpi(propagation_rule, 'overlays') && cnetwork.primary.args.overlays_max ~= 1
    if cnetwork.primary.args.overlays_max == 0
        coeff = max(max(cnetwork.primary.net));
    else
        coeff = cnetwork.primary.args.overlays_max;
    end
    real_density = real_density / coeff;
    if ~enable_guiding
        theoretical_error_rate = 1 - (1 - real_density)^(erasures*(l-1)+l*(Chi-c));
    else
        theoretical_error_rate = 1 - (1 - real_density)^(erasures*(l-1));
    end
elseif strcmpi(propagation_rule, 'overlays_ehsan2') && cnetwork.primary.args.overlays_max ~= 1
    theoretical_error_rate = 1-(1-real_density^(c-1))^c;
else
    if ~enable_guiding % different error rate when guided mask is enabled (and it's lower than blind decoding)
        %theoretical_error_rate = 1 - (1 - real_density^(c-erasures))^(erasures*(l-1)+l*(Chi-c)); % = spurious_cliques_proba. spurious cliques = nonvalid cliques that we did not memorize and which rests inopportunely on the edges of valid cliques, which we learned and want to remember. In other words: what is the probability of emergence of wrong cliques that we did not learn but which emerges from combinations of cliques we learned? This is influenced heavily by the density (higher density = more errors). Also, error rate is only per one iteration, if you use more iterations to converge the real error may be considerably lower. % NOTE: this is the correct error rate from the 2014 Behrooz paper but works only if concurrent_cliques = 1.
        theoretical_error_rate = 1 - binocdf(c-erasures-1, concurrent_cliques*(c-erasures), real_density)^(erasures*(l-1)+l*(Chi-c)); % generalization of the error rate given in the 2014 Behrooz paper, this works with any value of concurrent_cliques
    else
        %theoretical_error_rate = 1 - (1 - real_density^(c-erasures))^(erasures*(l-1)); % correct error rate from the 2014 Behrooz paper but works only if concurrent_cliques = 1.
        theoretical_error_rate = 1 - binocdf(c-erasures-1, concurrent_cliques*(c-erasures), real_density)^(erasures*(l-1)); % still the same generalization, but in guided mode so we count less many potentially spurious fanals (since we can exclude all clusters that the mask is excluding)
    end
    if concurrent_cliques > 1
        %theoretical_error_rate = 1-(1-theoretical_error_rate)^concurrent_cliques; WRONG, this just "doubles" the risk of spurious fanal for each concurrent clique, but it does not account for all possible combinations, you have to use a cumulative binomial distribution to count that!
        error_rate = 1 - (1 - error_rate)^(1/concurrent_cliques); % unbias approximation the real error rate by averaging, because statistically we exponentiate the error rate by the number of messages: error_rate^concurrent_cliques. Here we try to unbias that by finding the square root and thus to get error_rate per message, and not per concurrent_cliques messages as it is if biased. NOTE: remember that this is an approximation of the unbiased error rate, because we can't know which message caused the error, which would be the only way to get the exact error rate (so that we could know if, with for example concurrent_cliques = 2, if the final recovered mixed message is wrong because of 1 of the messages, or of both. Here we have no way to tell, so we suppose that in general, only one of the messages will fail but not both).
    end
end

%theoretical_error_correction_proba = 1 - theoretical_error_rate

% Compute euclidian error distance
error_distance = derr / (tests * concurrent_cliques * c * tampered_messages_per_test); % Euclidian distance: compute the esperance that a bit is wrongly flipped (has an incorrect value)

% Filling stats to return from function
test_stats = struct();
test_stats.real_density = real_density;
test_stats.error_rate = error_rate;
test_stats.theoretical_error_rate = theoretical_error_rate;
test_stats.error_distance = error_distance;
test_stats.similarity_measure = similarity_measure;
test_stats.matching_measure = matching_measure;

% Finally, show the error rate and some stats
if ~silent
    test_stats
    total_tampered_messages_tested = tests * tampered_messages_per_test
    aux.printcputime(cputime - totalperf, 'Total elapsed cpu time for test is %g seconds.\n'); aux.flushout();
    %c_optimal_approx = log(Chi*l/P0)/(2*(1-alpha)) % you have to define alpha = rate of errors per message you want to be able to correct ; P0 = probability or error = theoretical_error_rate you want
end

if ~silent; fprintf('=> Test done!\n'); aux.flushout(); end;

end % endfunction
