function [error_rate, theoretical_error_rate] = gbnn_test(network, sparsemessagestest, ...
                                                                                  l, c, Chi, ...
                                                                                  erasures, iterations, tampered_messages_per_test, tests, ...
                                                                                  enable_guiding, gamma_memory, threshold, propagation_rule, filtering_rule, tampering_type, ...
                                                                                  residual_memory, variable_length, concurrent_cliques, GWTA_first_iteration, GWTA_last_iteration, ...
                                                                                  silent)
%
% [error_rate, theoretical_error_rate] = gbnn_test(network, sparsemessagestest, ...
%                                                                                  l, c, Chi, ...
%                                                                                  erasures, iterations, tampered_messages_per_test, tests, ...
%                                                                                  enable_guiding, gamma_memory, threshold, propagation_rule, filtering_rule, tampering_type, ...
%                                                                                  residual_memory, variable_length, concurrent_cliques, GWTA_first_iteration, GWTA_last_iteration, ...
%                                                                                  silent)
%
% Feed a network and a matrix of sparse messages from which to pick samples for test, and this function will automatically sample some messages, tamper them, and then try to correct them. Finally, the error rate over all the processed messages will be returned.
%


% == Importing some useful functions
aux = gbnn_aux; % works with both MatLab and Octave

% == Init variables
if ~exist('Chi', 'var') || isempty(Chi)
    Chi = c;
end

if ~exist('gamma_memory', 'var') || isempty(gamma_memory)
    gamma_memory = 0;
end
if ~exist('threshold', 'var') || isempty(threshold)
    threshold = 0;
end
if ~exist('enable_guiding', 'var')
    enable_guiding = false;
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


% == Show vars (just for the record)
if ~silent
    % -- Network variables
    l
    c
    Chi % -- 2014 update

    % -- Test variables
    alpha = erasures / c
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
    variable_length
    concurrent_cliques
end



% == Init data structures and other vars - DO NOT TOUCH
sparse_cliques = true; % enable the creation of sparse cliques if Chi > c (cliques that don't use all available clusters but just c clusters per one message)
if Chi <= c
    Chi = c; % Chi can't be < c, thus here we ensure that
    sparse_cliques = false;
end
n = Chi * l; % total number of nodes ( = length of a message = total number of characters slots per message)

% Setup correct values for k (this is an automatic guess, but a manual value can be better depending on your dataset)
k = c*concurrent_cliques; % with propagation_rules GWTA and k-GWTA, usually we are looking to find at least as many winners as there are characters in the initial messages, which is at most c*concurrent_cliques (it can be less if the concurrent_cliques share some nodes, but this is unlikely if the density is low)
if strcmpi(filtering_rule, 'kWTA') || strcmpi(filtering_rule, 'kLKO') || strcmpi(filtering_rule, 'WsTA') % for all k local algorithms (k-WTA, k-LKO, WsTA, ...), k should be equal to the number of concurrent_cliques, since per cluster (remember that the rule here is local, thus per cluster) there is at most as many different characters per cluster as there are concurrent_cliques (since one clique can only use one node per cluster).
    k = concurrent_cliques;
end

mtest = size(sparsemessagestest, 1);

% -- A few error checks
if erasures > c
    error('Erasures > c which is not possible');
end


if ~silent; totalperf = cputime(); end; % for total time perfs

% #### Test phase
% == Run the test (error correction) and compute error rate (in reconstruction of original message without erasures)
if ~silent; fprintf('#### Testing phase (error correction of tampered messages with %i erasures)\n', erasures); aux.flushout(); end;
%sparsemessagestest = sparsemessages; % by default, use to test the same set as for learning
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
    inputm = sparsemessagestest(rndidx,:)'; % just fetch the messages and transpose them so that we have one sparsemessage per column (we don't generate them this way even if it's possible because of optimization: any, or, and sum are more efficient column-wise than row-wise, as any other MatLab/Octave function).
    init = inputm; % backup the original message before tampering, we will use the original to check if the network correctly corrected the erasure(s)

    %if ~debug; clear rndidx; end; % clear up memory - DEPRECATED because it violates the transparency (preventing the parfor loop to work)

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
        idxs = unidrnd([1 n], erasures, mconcat*tampered_messages_per_test); % generate random indices to be tampered
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
        % Mix up init
        init = reshape(init, n*tampered_messages_per_test, concurrent_cliques)';
        init = any(init); % mix up messages (by stacking concurrent_cliques messages side-by-side and then summing/anying them)
        %init = reshape(init, tampered_messages_per_test, n)'; % WRONG % unstack the messages vector into a matrix with one mixed sparsemessage per column
        init = reshape(init', n, tampered_messages_per_test); % unstack the messages vector into a matrix with one mixed sparsemessage per column

        % Mix up the tampered messages
        inputm = reshape(inputm, n*tampered_messages_per_test, concurrent_cliques)';
        inputm = any(inputm);
        inputm = reshape(inputm', n, tampered_messages_per_test);
    end

    if ~silent; aux.printtime(toc()); end;

    if ~silent; tperf = cputime(); end;
    % -- Prediction step: feed the tampered message to the network and wait for it to converge to a stable state, hopefully the corrected message.
    %if ~silent; fprintf('-- Feed to network and wait for convergence\n'); aux.flushout(); end;
    guiding_mask = [];
    if enable_guiding % if enabled, prepare the guiding mask (the list of clusters that we will keep, all the other nodes from other clusters will be set to 0). This guiding mask can be defined manually if you want, here to do it automatically we compute it from the initial untampered messages, thus we will keep nodes activated only in the clusters where there were activated nodes in the initial message.
        guiding_mask = any(reshape(init, l, tampered_messages_per_test * Chi)); % any is better than sum in our case, and it's also faster and keeps the logical datatype!
    end

    inputm = gbnn_correct(network, inputm, ...
                              l, c, Chi, ...
                              iterations, ...
                              k, guiding_mask, gamma_memory, threshold, propagation_rule, filtering_rule, tampering_type, ...
                              residual_memory, variable_length, concurrent_cliques, GWTA_first_iteration, GWTA_last_iteration, ...
                              silent);

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
    else
        %err = err + ~isequal(init,inputm);
        err = err + any(init ~= inputm); % remove the useless sum(min()) when we only have one message to compute the error from, this cuts the time by almost half
    end
    if ~silent; aux.printtime(toc()); end;

end

% Finally, show the error rate and some stats
error_rate = err / (tests * tampered_messages_per_test);
if ~silent
    real_density = full(  (sum(sum(network)) - sum(diag(network))) / (Chi*(Chi-1) * l^2)  )
    theoretical_error_rate = -1;
    if enable_guiding % different error rate when guided mask is enabled (and it's lower than blind decoding)
        theoretical_error_rate = 1 - (1 - real_density^(c-erasures))^(erasures*(l-1))
    else
        theoretical_error_rate = 1 - (1 - real_density^(c-erasures))^(erasures*(l-1)+l*(Chi-c)) % = spurious_cliques_proba. spurious cliques = nonvalid cliques that we did not memorize and which rests inopportunely on the edges of valid cliques, which we learned and want to remember. In other words: what is the probability of emergence of wrong cliques that we did not learn but which emerges from combinations of cliques we learned? This is influenced heavily by the density (higher density = more errors). Also, error rate is only per one iteration, if you use more iterations to converge the real error may be considerably lower. % TODO: does not compute the correct theoretical error rate if concurrent_cliques > 1.
    end
    %theoretical_error_correction_proba = 1 - spurious_cliques_proba
    error_rate
    total_tampered_messages_tested = tests * tampered_messages_per_test
    aux.printcputime(cputime - totalperf, 'Total elapsed cpu time for test is %g seconds.\n'); aux.flushout();
    %c_optimal_approx = log(Chi*l/P0)/(2*(1-alpha)) % you have to define alpha = rate of errors per message you want to be able to correct ; P0 = probability or error = theoretical_error_rate you want
end

if ~silent; fprintf('=> Test done!\n'); aux.flushout(); end;

end % endfunction
