function [partial_messages, propag] = gbnn_correct(varargin)
%
% partial_messages = gbnn_correct(cnetwork, partial_messages, ...
%                                                                                  iterations, ...
%                                                                                  k, guiding_mask, gamma_memory, threshold, propagation_rule, filtering_rule, ...
%                                                                                  residual_memory, concurrent_cliques, concurrent_disequilibrium, filtering_rule_first_iteration, filtering_rule_last_iteration, ...
%                                                                                  enable_overlays, overlays_max, overlays_interpolation, ...
%                                                                                  silent)
%
% Feed a network and partially tampered messages, and will let the network try to 'remember' a message that corresponds to the given input. The function will return the recovered message(s) (but no error rate, see gbnn_test.m for this purpose).
%
% concurrent_disequilibrium is a disambiguation trick in the concurrent case, it helps a great deal with the performance by trying to find only one clique at a time (thus this will multiply the number of iterations: if iterations == 4 and concurrent_cliques == 3, you will have a total 4*3=12 iterations if you enable concurrent_disequilibrium), but that's a small price to pay! Also note that you need to set iterations > 1 to get any beneficial effect from the disequilibrium (iterations = 4 is good).
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
    'concurrent_cliques', 1, ... % 1 is disabled, > 1 enables and specify the number of concurrent messages/cliques to decode concurrently
    'concurrent_disequilibrium', false, ...
    'filtering_rule_first_iteration', false, ...
    'filtering_rule_last_iteration', false, ...
    'k', 0, ...
    'enable_dropconnect', false, ...
    'dropconnect_p', 0.5, ...
    ...
    ... % Overlays / Tags extension
    'enable_overlays', false, ...
    'overlays_max', 0, ...
    'overlays_interpolation', 'uniform', ...
    ...
    ... % Internal variables (automatically provided by gbnn_test(), you should not modify this on your own
    'cnetwork_choose', 'primary', ... % if multiple networks are available, specify which one to use for the propagation (useful for auxiliary support network)
    ...
    ... % Debug stuffs
    'silent', false);

% Process the arguments
arguments = aux.getnargs(varargin, arguments_defaults, true);

% Load variables into local namespace (called workspace in MatLab)
aux.varspull(arguments);
aux.varspull(cnetwork.(cnetwork_choose).args);

% == Sanity Checks
if isempty(cnetwork) || isempty(partial_messages) || isfield(cnetwork.(cnetwork_choose).args, 'l') == 0 || isfield(cnetwork.(cnetwork_choose).args, 'c') == 0
    error('Missing arguments: cnetwork, partial_messages, cnetwork.%s.args.l and cnetwork.%s.args.c are mandatory!', cnetwork_choose, cnetwork_choose);
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
if enable_overlays && (islogical(cnetwork.primary.net) || max(max(cnetwork.primary.net)) == 1)
    error('cannot use overlays because overlays were not learned. Please first use gbnn_learn() with argument enable_overlays = true');
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
    if ~silent; printf('Notice: provided messages is not thrifty. Converting automatically to thrifty code.\n'); end;
    partial_messages = gbnn_messages2thrifty(partial_messages, l);
end

mpartial = size(partial_messages, 2); % mpartial = number of messages to reconstruct. This is also equal to tampered_messages_per_test in gbnn_test.m

% Setup correct values for k (this is an automatic guess, but a manual value can be better depending on your dataset)
if k < 1
    k = c*concurrent_cliques; % with propagation_rules GWTA and k-GWTA, usually we are looking to find at least as many winners as there are characters in the initial messages, which is at most c*concurrent_cliques (it can be less if the concurrent_cliques share some nodes, but this is unlikely if the density is low)
    if concurrent_cliques > 1 && concurrent_disequilibrium;  k = c; end;
    if strcmpi(filtering_rule, 'kWTA') || strcmpi(filtering_rule, 'kLKO') || strcmpi(filtering_rule, 'WsTA') % for all k local algorithms (k-WTA, k-LKO, WsTA, ...), k should be equal to the number of concurrent_cliques, since per cluster (remember that the rule here is local, thus per cluster) there is at most as many different characters per cluster as there are concurrent_cliques (since one clique can only use one node per cluster).
        k = concurrent_cliques;
        if concurrent_cliques > 1 && concurrent_disequilibrium;  k = 1; end;
    end
end

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


% Select the network on which we will work
net = cnetwork.(cnetwork_choose).net;
netargs = cnetwork.(cnetwork_choose).args;

% Overlays reduction preprocessing: if enabled, we reduce the number of overlays to a fixed number. To do this, we interpolate the messages ids inside the network by various methods.
% The goal is to preserve the ids but assign them a new id (which will maybe merge them with other messages, but at least one message isn't broken into parts, see 'uniform' comment for more details).
% Note: we reduce the number of tags at correction and not at learning just for efficiency sake in experiments: this allows us to learn only one network and then try any number of tags we want with any method without having to recompute the network everytime. Of course, in a real setting, you should do the reduction just after the learning (or even during) to reduce the prediction/correction step footprint (at the cost of a longer learning step).
if enable_overlays
    if overlays_max > 0 % If overlays_max == 0 then we use all overlays
        % Modulo reduction: use a sort of roulette to assign new overlay ids. Eg: for overlays_max == 3, the network [1 2 3 ; 4 5 6] will be reduced to [1 2 3 ; 1 2 3]
        if strcmpi(overlays_interpolation, 'mod')
            net = spfun(@(x) mod(x-1, overlays_max)+1, net);
        % Renormalization reduction: renormalize all overlays into the reduced range, but preserve their order (old messages will still get the lowest numbers and most recent messages will have highest). Eg: for overlays_max == 3, the network [1 2 3 ; 4 5 6] will be reduced to [1 1 2 ; 2 3 3]
        elseif strcmpi(overlays_interpolation, 'norm')
            maxa = max(nonzeros(net));
            mina = min(nonzeros(net));
            net = ceil(spfun(@(x) ((x-(mina-1)) / (maxa-(mina-1))), net) * overlays_max); % note: we use ceil to make sure that we don't lose any fanal because of rounding (near 0 isn't 0 but is an activable fanal, thus we should not set it to 0)
        % Random uniform reduction: reassign a random id to every message, in the range of overlays_max. Eg: for overlays_max == 3, the network [1 1 3 3 ; 5 5 7 7] _can_ be reduced to [2 2 1 1 ; 2 2 3 3] or to any other randomly picked set of reassignment. Note that all messages with the same id will be reassigned to the same id (eg: [1 1 1] can be reassigned to [3 3 3] but NOT to [1 2 3] because this breaks the message into parts instead of preserving it).
        elseif strcmpi(overlays_interpolation, 'uniform')
            maxa = max(nonzeros(net));
            random_map = randi(overlays_max, maxa, 1);
            % net = spfun(@(x) random_map(x), net); % SLOWER than direct indexing!
            net(net > 0) = random_map(nonzeros(net)); % faster!
        end
    end
end

% -- A few more preparations just before starting the correction loop

% Backup the full network if we use dropconnect, because we will randomly erase connections at each iteration
if enable_dropconnect
    orig_net = net;
end

% Backup the initial messages if we use disequilibrium trick in concurrent case because we will erase one fanal at each diter
diterations = 1;
concurrent_cliques_bak = concurrent_cliques;
if concurrent_cliques > 1 && concurrent_disequilibrium
    diterations = concurrent_cliques_bak;
    partial_messages_bak = partial_messages;
    out_final = logical(sparse(size(partial_messages,1), size(partial_messages,2)));
    concurrent_cliques = 1;
end

% #### Correction phase
for diter=1:diterations
    % Disequilibrium trick pre-processing: we try to disequilibrate the message and thus decode only one clique at a time (the goal is that one clique will get the upper hand by either superboosting one fanal score and thus one clique overall score, or by erasing one fanal so that one clique gets a lower score).
    % Idea from Xiaoran Jiang, thank's a lot!
    % concurrent_disequilibrium = 1 for superscore mode, 2 for one fanal erasure, 3 for nothing at all just trying to decode one clique at a time without any trick
    if concurrent_cliques_bak > 1 && concurrent_disequilibrium && diter < diterations % do not erase nor superboost a fanal at the last iteration, because we already erased all the other cliques thus we don't need to disequilibrate at the last step
        if concurrent_disequilibrium ~= 3 % third disequilibrium technique: we don't do anything, we will just try to find only one clique at one time, but without doing any special trick
            % get the number of activated fanals per message
            franges = sum(partial_messages);
            % select a random fanal to erase per message
            random_idxs = ceil(franges .* rand(1, numel(franges)));
            % adjust offset to get matlab style indexes (instead of per column index, we get indexes counting from 1 at the start of the matrix to numel at the end of the matrix)
            franges = cumsum(franges);
            franges = [0 franges(1:end-1)];
            random_idxs = random_idxs + franges;
            % Find which fanal we will select per message
            idxs = find(partial_messages);
            diseq_idxs = idxs(full(random_idxs));

            % Erase those fanals
            % NOTE: this does not work if the cliques are heavily overlapping, because erasing one fanal will probably erase a shared fanal and thus there won't be any disequilibrium. But anyway if the cliques are heavily overlapping, this means that the density is super high and anyway we can't do anything about the error rate.
            % NOTE2: tried to do the generalization trick proposed by Xiaoran, but it doesn't work well: at the end of one iteration, instead of considering that the decoded message is one clique, consider that the decoded message is multiple cliques, and instead of keeping that, keep the fanals that were activated in the original message but are now shutdown in the decoded message, and remove these fanals from the original message to now try to find the other cliques the same way
            % NOTE3: this IS working for multiple cliques > 2, I have no idea why but it works (albeit with a higher error rate than with trick 1 superboost score or 3 do nothing)
            if concurrent_disequilibrium == 2
                partial_messages(diseq_idxs) = 0;
            % Superboost the score of one fanal to give advantage to one and only one clique for now (because this fanal will propagate its score to one clique)
            else
                partial_messages = double(partial_messages); % must convert to double so that we can set an integer (non binary/logical) value
                %partial_messages(diseq_idxs) = sum(partial_messages);
                partial_messages(diseq_idxs) = concurrent_cliques_bak*c + 1;
            end

            % Clear memory
            clear random_idxs;
            clear idxs;
            clear franges;
        end
    end
    % Printing info on concurrent disequilibrium trick
    if concurrent_cliques_bak > 1 && concurrent_disequilibrium
        if ~silent; printf('--> Finding clique %i by disequilibrium type %i\n', diter, concurrent_disequilibrium); aux.flushout(); end;
    end

    for iter=1:iterations % To let the network converge towards a stable state...
        if ~silent
            fprintf('-- Propagation iteration %i\n', iter); aux.flushout();
            tic();
        end

        % -- Some preprocessing
        if enable_dropconnect % DropConnect
            net = dropconnect(orig_net, dropconnect_p);
        end

        %if concurrent_disequilibrium == 1 && ...
        %(strcmpi(filtering_rule, 'GWSTA-ML') || ...
          %(filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'gwsta-ml') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'gwsta-ml') && iter == iterations) )
            %partial_messages = double(partial_messages);
            %partial_messages(diseq_idxs) = concurrent_cliques_bak*c;
        %end

        % 1- Update the network's state: Push message and propagate through the network
        % NOTE: this is the CPU bottleneck if you use many messages with c > 8

        % Propagate on the primary network
        % -- Vectorized version - fastest and low memory overhead
        % Sum-of-Sum: we simply compute per node the sum of all incoming activated edges
        if strcmpi(propagation_rule, 'sum') || (enable_overlays && (strcmpi(propagation_rule, 'overlays_old_filter') || strcmpi(propagation_rule, 'overlays_old_filter2')))
            % We use the standard way to compute the propagation in an adjacency matrix: by matrix multiplication
            % Sum-of-Sum / Matrix multiplication is the same as computing the in-degree(a) for each node a, since each connected and activated link to a is equivalent to +1, thus this is equivalent to the total number of connected and activated link to a.
            % This operation may seem daunting but in fact it's quite simple and just does what it sounds like: for each message (row in the transposed partial_messages matrix = in fanal), we check if any of its activated fanal is connect to any of the network's fanal (1 out fanal = 1 column of the net's matrix). The result is the sum or count of the number of in connections for each of the network's fanals.
            % Or you can see the network as a black box: it simply transforms one list of messages into another format conditionally on the network's links.
            % NOTE: when enable_overlays is activated, we doesn't care here because we won't use tags for propagation (we must binarize the network, hence the logical(net))
            if aux.isOctave()
                propag = (partial_messages' * logical(net))'; % Propagate the previous message state by using a simple matrix product (equivalent to similarity/likelihood? or is it more like a markov chain convergence?). Eg: if network = [0 1 0 ; 1 0 0], partial_messages = [1 1 0] will match quite well: [1 1 0] while partial_messages = [0 0 1] will not match at all: [0 0 0]
                % WRONG: net * partial_messages : this works only with non-bridge networks (eg: primary, auxiliary, but not with prim2auxnet !)
            else % MatLab cannot do matrix multiplication on logical matrices...
                propag = (double(partial_messages') * double(logical(net)))';
            end
        % Overlays global propagation: compute the mode-of-products and then the sum-of-equalities with the mode. The goal is to activate all fanals in the network which are in the input message, then extract all the edges that will be activated, then look at their tags id and keep the major one (by using a mode). Then we filter all the other edges except the ones having this tag. End of global overlays propagation.
        % Note the heavy usage of the generalized matrix multiplication.
        elseif enable_overlays && strcmpi(propagation_rule, 'overlays_old')
            fastmode = @(x) max(aux.fastmode(nonzeros(x))); % prepare the fastmode function (in case of draw between two values which are both the mode, keep the maximum one = most recent one). It will work globally: we find the mode for all edges weights in the network.
            winner_overlays = gmtimes(double(partial_messages'), net, fastmode, @times, [mpartial, 1])'; % compute the mode-of-products to fetch the major tag per message (or rather: per network)
            partial_messages = partial_messages .* bsxfun(@times, double(partial_messages), winner_overlays); % prepare the input messages with the mode (instead of binary message, each message will be assigned the id of its major tag)
            propag = gmtimes(partial_messages', net, @sum, @(a,b) and(full(a>0), bsxfun(@eq, a, b))); % finally, compute the sum-of-equalities: we activate only the edges corresponding to the major tag for this message.
            %@(a,b) and(full(a > 0), eq(full(a),full(b)))
            propag = propag';
        % Else error, the propagation_rule does not exist
        else
            error('Unrecognized propagation_rule: %s', propagation_rule);
        % TODO: sum-of-max use spfun() or arrayfun() to do a custom matrix computation per column: first an and, then a reshape to get only nodes per cluster on one column, then any, then we have our new message! Or compute sos then diff(sos - som) then sos - diff(sos-sum)
        % TODO: Sum-of-Max = for each node, compute the sum of incoming link per cluster (thus if a node is connected to 4 other nodes, but 3 of them belong to the same cluster, the score will be 2 = 1 for the node alone in its cluster + 1 for the other 3 belonging to the same cluster).
        % TODO: Normalized Sum-of-Sum = for each node, compute the sum of sum but divide the weight=1 of each incoming link by the number of activated node in the cluster where the link points to (thus if a node is connected to 4 other nodes, but 3 of them belong to the same cluster, the score will be 1 + 1/3 + 1/3 + 1/3 = 2). The score will be different than Sum of Max if concurrent_cliques is enabled (because then the number of activated nodes can be higher and divide more the incoming scores).
        end

        % Propagate in parallel on the auxiliary network
        mes_echo = [];
        if isfield(cnetwork, 'auxiliary') && strcmpi(cnetwork_choose, 'primary')

            if ~silent
                fprintf('Propagating through auxiliary network...\n'); aux.flushout();
                auxpropagtime = cputime();
            end

            prim2auxnet = cnetwork.auxiliary.prim2auxnet;
            if cnetwork.auxiliary.args.enable_dropconnect
                prim2auxnet = dropconnect(prim2auxnet, cnetwork.auxiliary.args.dropconnect_p);
            end

            % Echo from primary to auxiliary
            mes_echo = partial_messages;
            if aux.isOctave()
                mes_echo = (mes_echo' * prim2auxnet)';
            else
                mes_echo = (double(mes_echo') * double(prim2auxnet))';
            end
            % FILTERING -1: binarize the echo: WRONG!
            %mes_echo = logical(mes_echo); % NEVER just binarize the echo: this will give as much power to spurious auxiliary fanals as to correct auxiliary fanals! Because it's quite rare that all fanals of a primary clique are all linked exclusively to one auxiliary clique, you have great chances that at least one primary fanal will trigger two different auxiliary cliques, and thus produce confusion. By binarizing, all auxiliary cliques will have an equal weight, which is false because false auxiliary cliques have a lower score before binarizing, and we should take that into account!
            % FILTERING 0: do nothing! Just filter at the echo back!
            % FILTERING 1
            %mes_echo = bsxfun(@eq, mes_echo, max(mes_echo)); % GWTA: keep only the max values
            % FILTERING 1.5 : keep max but keep their original values
            mes_echo = bsxfun(@eq, mes_echo, max(mes_echo));
            mes_echo = bsxfun(@times, mes_echo, max(mes_echo));
            % FILTERING 2
            %mes_echo(mes_echo < c) = 0;
            %mes_echo = logical(mes_echo);
            % FILTERING 3: inhibition
            %mes_echo(mes_echo >= cnetwork.auxiliary.args.c) = 0;
            %mes_echo = logical(mes_echo);

            % Propagate through the auxiliary network to find the correct clique
            if ~isempty(cnetwork.auxiliary.net)
                %mes_echo = logical(mes_echo); % should be logical else it will be converted into thrifty code by gbnn_messages2thrifty.m automatically!
                mes_echo = bsxfun(@eq, mes_echo, max(mes_echo)); % another way of converting into logical but without losing all infos (should be logical else it will be converted into thrifty code by gbnn_messages2thrifty.m automatically!)
                %mes_echo = gbnn_correct('cnetwork', cnetwork, 'partial_messages', mes_echo, 'cnetwork_choose', 'auxiliary', 'filtering_rule', 'gwsta', 'enable_dropconnect', cnetwork.auxiliary.args.enable_dropconnect, 'dropconnect_p', cnetwork.auxiliary.args.dropconnect_p, 'silent', true); % with dropconnect
                mes_echo = gbnn_correct('cnetwork', cnetwork, 'partial_messages', mes_echo, 'cnetwork_choose', 'auxiliary', 'filtering_rule', 'gwsta', 'silent', true); % without dropconnect
            end

            prim2auxnet = cnetwork.auxiliary.prim2auxnet;
            if cnetwork.auxiliary.args.enable_dropconnect
                prim2auxnet = dropconnect(prim2auxnet, cnetwork.auxiliary.args.dropconnect_p);
            end

            % Echo back from auxiliary to primary
            if aux.isOctave()
                mes_echo = (mes_echo' * prim2auxnet')';
            else
                mes_echo = (double(mes_echo') * double(prim2auxnet'))';
            end
            % FILTERING -1: binarize
            %mes_echo = logical(mes_echo);
            % FILTERING 1
            %mes_echo = bsxfun(@eq, mes_echo, max(mes_echo));
            % FILTERING 1.5 : keep max but keep their original values
            %mes_echo = bsxfun(@eq, mes_echo, max(mes_echo));
            %mes_echo = bsxfun(@times, mes_echo, max(mes_echo));
            % FILTERING 2
            mes_echo(mes_echo < cnetwork.auxiliary.args.c) = 0;
            mes_echo = logical(mes_echo);
            % FILTERING 3: inhibition
            %mes_echo(mes_echo >= cnetwork.auxiliary.args.c) = 0;
            %mes_echo = logical(mes_echo);
            
            % BEST COMBINATION: 1.5 + 2

            %propag = propag + mes_echo; % auxiliary network echo
            %propag = propag - mes_echo; % inhibition
            %propag = propag - ~mes_echo; % inhibition of non-connected fanals

            if ~silent
                aux.printcputime(cputime - auxpropagtime, 'Elapsed cpu time for auxiliary propagation is %g seconds.\n'); aux.flushout();
            end
        end

        % 2- In-between processing (just after propagation but before filtering)
        
        % Gamma memory
        if gamma_memory > 0
            propag = propag + (gamma_memory .* partial_messages); % memory effect: keep a bit of the previous nodes scores.
            % NOTE: gamma memory is equivalent as a loop on all fanals. Thus, you can also set gamma on the cnetwork diagonal before propagation: net(1:n:end) = net(1:n:end) * gamma_memory.
            % NOTE2: always set gamma = 1 this will give you the best performances, because when gamma == 0, fanals initially activated will have a lower score than others, because they can't stimulate themselves. Eg: if fanals A and B are connected to C and D, and if A and B are initially activated, the scores with gamma = 0 will be as follows: A:1, B:1, C:2, D:2 because A is only stimulated by B and inversely for B, but C and D are stimulated by both A and B. Thus after one interation, we lose A and B, which were initially activated! However if we set gamma = 1, we get A:2 B:2 C:2 D:2 because A is stimulated by B and by A itself.
            % NOTE3: setting gamma > 1 will force the initially activated fanals to always be activated. Can be good when you use erasures, but not good at all if you have noise.
            % NOTE4: here in this implementation, gamma_memory is already == 1.
        end

        % Activation threshold: set to 0 all nodes with a score lower than the activation threshold (ie: those nodes won't emit a spike)
        if threshold > 0
            propag(propag < threshold) = 0;
        end;

        % Guiding mask
        if ~isempty(guiding_mask) % Apply guiding mask to filter out useless clusters. TODO: the filtering should be directly inside the propagation rule at the moment of the matrix product to avoid useless computations (but this is difficult to do this in MatLab in a vectorized way...).
            propag = reshape(propag, l, mpartial * Chi);
            if ~aux.isOctave()
                propag = bsxfun(@times, propag, guiding_mask);
            else
                propag = sparse(bsxfun(@times, propag, double(guiding_mask)));
            end
            propag = reshape(propag, n, mpartial);
        end


        % 3- Filtering rules aka activation rules (apply a rule to filter out useless/interfering nodes)

        % -- Vectorized versions - fastest!
        out = logical(sparse(size(partial_messages,1), size(partial_messages,2))); % empty binary sparse matrix, it will later store the next network state after winner-takes-all is applied
        % Do nothing, useful for auxiliary network since it's just a reverberation
        if strcmpi(filtering_rule, 'none') || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'none') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'none') && iter == iterations)
            out = propag;
        % Just make sure the values are binary
        elseif strcmpi(filtering_rule, 'binary') || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'binary') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'binary') && iter == iterations)
            out = logical(propag);
        % Winner-take-all : per cluster, keep only the maximum score node active (if multiple nodes share the max score, we keep them all activated). Thus the WTA is based on score value, contrarywise to k-WTA which is based on the number of active node k.
        elseif strcmpi(filtering_rule, 'wta') || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'wta') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'wta') && iter == iterations)
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
        elseif strcmpi(filtering_rule, 'kwta') || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'kwta') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'kwta') && iter == iterations)
            propag = reshape(propag, l, mpartial * Chi); % reshape so that we can do the WTA by a simple column-wise WTA (and it's efficient in MatLab since matrices - and even more with sparse matrices - are stored as column vectors, thus it efficiently use the memory cache since this is the most limiting factor above CPU power). See also: Locality of reference.
            [~, idxs] = sort(propag, 'descend');
            idxs = bsxfun( @plus, idxs, 0:l:((mpartial * n)-1) );
            [I, J] = ind2sub(size(propag), idxs(1:k, :));
            out = logical(sparse(I, J, 1, l, mpartial * Chi));
            out = and(propag, out); % IMPORTANT NO FALSE WINNER TRICK: if sparse_cliques or variable_length, filter out winning nodes with 0 score (selected because there's no other node with any score in this cluster, see above in filtering_rule = 0)
            out = reshape(out, n, mpartial);
        % One Global Winner-take-all: only keep one value, but at the last iteration keep them all
        elseif (strcmpi(filtering_rule, 'ogwta') && iter < iterations) || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'ogwta') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'ogwta') && iter == iterations)
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
        elseif strcmpi(filtering_rule, 'gwta') || (strcmpi(filtering_rule, 'ogwta') && iter == iterations) || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'gwta') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'gwta') && iter == iterations)
            winner_vals = max(propag); % get global max scores for each message
            if ~aux.isOctave()
                out = logical(bsxfun(@eq, winner_vals, propag));
            else
                out = logical(sparse(bsxfun(@eq, winner_vals, propag)));
            end
            out = and(propag, out); % No false winner trick
        % Global k-Winners-take-all: keep the best k first nodes having the maximum score over the whole message (same as k-WTA but at the message level instead of per-cluster).
        elseif strcmpi(filtering_rule, 'GkWTA') || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'gkwta') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'gkwta') && iter == iterations)
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
        % Intuitively, GWsTA is better than GWTA because in case of equal score between fanals, this means there is an ambiguity, and while GWTA will randomly cut off some fanals without any clue, GWsTA will say "well, okay, I don't know which one to choose, I will keep them all and decide at a later iteration, hoping that the next propagation will make things clearer".
        elseif strcmpi(filtering_rule, 'WsTA') || strcmpi(filtering_rule, 'GWsTA') || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'gwsta') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'gwsta') && iter == iterations)
            % Local WinnerS-take-all: Reshape to get only one cluster per column (instead of one message per column)
            if strcmpi(filtering_rule, 'WsTA')
                propag = reshape(propag, l, mpartial * Chi);
            end

            max_scores = sort(propag,'descend'); % sort scores
            kmax_score = max_scores(k, :); % find the kth max score
            kmax_score(kmax_score == 0) = realmin(); % No false winner trick: where 0 is the value of the kth winner, we replace the 0 value by the minimum real value (greater than 0) so that when we filter out fanals with a score below the kth winner, we always filter out 0 (even if the kth winner had value 0). For GWsTA, this is a better version of the no false winner trick because we replace 0 winner scores by the minimum real value, thus after we will still get a sparse matrix (else if we do the trick only after the bsxfun, we will get a matrix filled by 1 where there are 0 and the winner score is 0, which is a lot of memory used for nothing).
            out = logical(bsxfun(@ge, propag, kmax_score)); % filter out any value below the kth max score (and since 0 was replaced by realmin, we also filter out 0 values, they won't get activated because of the bsxfun)
            if aux.isOctave(); out = sparse(out); end; % Octave's bsxfun breaks the sparsity...
            out = and(propag, out); % No false winner trick: avoids that if the kth max score is in fact 0, we choose 0 as activating score (0 scoring nodes will be activated, which is completely wrong!). Here we check against the original propagation matrix: if the node wasn't activated then, it shouldn't be now after filtering.

            % Local WinnerS-take-all: Reshape back the clusters into messages
            if strcmpi(filtering_rule, 'WsTA')
                out = reshape(out, n, mpartial);
            end

        % Loser-kicked-out (locally, per cluster, we kick loser nodes with min score except if min == max of this cluster).
        % Global Loser-Kicked-Out (deactivate all nodes with min score in the message)
        % Both are implemented the same way, the only difference is that for global we process losers per message, and with local we process them per cluster
        elseif strcmpi(filtering_rule, 'LKO') || strcmpi(filtering_rule, 'GLKO') || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'glko') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'glko') && iter == iterations)
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
                    strcmpi(filtering_rule, 'oLKO') || strcmpi(filtering_rule, 'oGLKO') || ...
                    (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'gklko') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'gklko') && iter == iterations)
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
        % Note: doesn't perform well.
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
        % Exhaustive GWsTA, as used in the Maximum Likelihood as a prefilter.
        % This is a kind of analytical GWsTA: we don't select based on score but rather we remove all nodes that we are certain they have a score too low to be part of the clique we are looking for. So here this is a simple constant threshold whereas GWsTA is based on finding c fanals with best scores.
        % Thus, this method does not remove correct fanals (it always has a matching_score of 1 if iterations == 1), while GWsTA may remove correct fanals if some spurious fanal has a very high score (ie: linked to fanals in both cliques).
        % Biologically, we could say that this is kind of a threshold: the system knows that we are looking for cliques of length c, thus we know what is the minimum score (c-erasures).
        % NOTE: when concurrent_cliques == 1, GWsTA-ML is equivalent to GWsTA.
        elseif strcmpi(filtering_rule, 'GWsTA-ML') || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'gwsta-ml') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'gwsta-ml') && iter == iterations)
            % Try to heuristically find the number of erasures
            erasures = c - (mode(sum(partial_messages)) / concurrent_cliques); % simple standard case
            if concurrent_disequilibrium && exist('concurrent_cliques_bak', 'var')
                erasures = c - (mode(sum(logical(partial_messages))) / concurrent_cliques_bak); % special case when using concurrent disequilibrium trick
            end

            % Filter out all useless nodes (ie: if they have a score below the length of a message, then these nodes can't possibly be part of a clique of length c, which is what we are looking for)
            if concurrent_disequilibrium == 1
                propag(propag < ((c-erasures-1) + concurrent_cliques_bak * c)) = 0; % special case when using first type of concurrent disequilibrium trick: we must add the superboost score and we need to remove 1 because c-erasures counts all initially activated fanals, but we superboosted one so we should remove its 1 score and we add its superboosted score.
            else
                propag(propag < (c-erasures)) = 0; % simple standard case
            end

            % Binarize
            out = logical(propag);

        % Maximum likelihood or Exhaustive search (Depth-First Search or Breadth-First Search)
        % This will find the k-clique where k = c, in other words it will reconstruct one node at a time a clique of order c, and if it can't find such a clique, then it returns an empty message (we failed, there's no clique). If there's a clique of length c, this algorithm will find it and return it. If there are multiple cliques of length c, this algorithm will find just one and returns it, whether it's the one we are looking for or not (thus it does not handly ambiguity).
        % How it works: we consider that we are in a graph, where every node in this graph corresponds to a submessage containing a subset of the nodes of the full message. In the ascending approach, the first node is totally empty, and each subnodes has one fanal, then each subsubnodes has two fanals, then each subsubsubnodes has three fanals, etc... until at the end we get nodes where the messages contain c fanals (we can go deeper but we stop at this level since we are trying to find a kclique and not the maximum clique). Thus we can consider this problem to be a simple tree search by walking, thus as Depth-First Search or Breadth-First Search or even AStar (also called Branch and Bound, and this is what is used to find maximal clique usually).
        % Finding a k-clique is a NP-hard problem, thus this algorithm will be slow, but it's enhanced with tricks from Constraints Programming thus as backtracking and domain-elimination (if a combination of node A-B-C does not form a clique, then we avoid exploring this subtree entirely, meaning that we will avoid A-B-C-D, A-B-C-E, A-B-C-D-E, etc.).
        % Note that this is the only filtering algorithm that isn't a heuristic, since it does not use scores at all, but rather try all combinations of nodes until it finds a clique of order c. Thus, this algorithm does not suffer from spurious fanals (which have a score above correct fanals), but only suffers from spurious cliques (for this algorithm to be confused, there must be an ambiguity = two cliques of order c, so to have an ambiguity, a spurious fanal must be connected to at least (c-1) correct fanals so that it really forms a clique).
        % TODO: try to use a SAT solver, similarly to the knapsack problem? This should be a lot faster than doing it manually. NO: rather use efficient algorithms specifically to find maximum cliques, like Constraint Programming Régin, J. C. (2003, January). Using constraint programming to solve the maximum clique problem. In Principles and Practice of Constraint Programming–CP 2003 (pp. 634-648). Springer Berlin Heidelberg.
        % or MCS or BBMC (BB-MaxClique): http://arxiv.org/pdf/1207.4616.pdf Pablo San Segundo, Diego Rodr´iguez-Losada, and August´in Jim´enez. An exact bit-parallel algorithm for the maximum clique problem. Computers and Operations Research, 38:571–581, 2011.
        % See also: https://www.cs.purdue.edu/homes/agebreme/publications/fastClq-WAW13.pdf
        % NOTE: on performances: one k-clique is easy enough to find and quite quick, but it's a lot harder to find multiple k-cliques, because the algorithm might get stuck in the subtree leading to only the k-clique we already found, the other k-cliques being on different subtrees altogether. To avoid this, there is a re-sort after each k-clique is just found (see just_found variable), which allows to explore altogether different subtrees as a first guess. However, this brings another harder (and hardest) case: when the different k-cliques overlap, you have to explore in the middle of the tree, and this is the hardest case since you have fanals from the already found k-clique overlapping with the next k-clique to find, thus we can't just explore an altogether different subtree, but rather the middle of the tree which is a mix between the new k-clique and the already found one.
        % TODO: try to use a genetic algorithm? Or a cuckoo search?
        elseif strcmpi(filtering_rule, 'ML') || strcmpi(filtering_rule, 'DFS') || strcmpi(filtering_rule, 'BFS') || strcmpi(filtering_rule, 'MLD') || ...
        (filtering_rule_first_iteration && strcmpi(filtering_rule_first_iteration, 'ml') && iter == 1) || (filtering_rule_last_iteration && strcmpi(filtering_rule_last_iteration, 'ml') && iter == iterations)
            erasures = c - (mode(sum(partial_messages)) / concurrent_cliques); % try to heuristically find the number of erasures
            propag(propag < (c-erasures)) = 0; % Filter out all useless nodes (ie: if they have a score below the length of a message, then these nodes can't possibly be part of a clique of length c, which is what we are looking for). This is almost like GWsTA but not exactly: GWsTA may remove correct fanals because they have a score lower than the c highest, even if the correct fanals have a score >= (c-erasures). This is confirmed by the matching_measure (compare with GWsTA-ML which is exactly the same procedure as this one here).
            out = logical(sparse(size(propag,1), size(propag,2))); % Init the output messages. By default if we can't find a k-clique, we will set the message to all 0's
            % Presetting the search mode into a simple boolean, it will speed things up instead of comparing strings everytime inside the loop
            mode_search = 0;
            % Depth-First Search: we first explore the sub nodes of the current node then later we will explore sibling nodes of current node
            if strcmpi(filtering_rule, 'ML') || strcmpi(filtering_rule, 'DFS')
                mode_search = 0;
            % Breadth-First Search: we first explore sibling nodes (nodes at the same level as current node) and then after we will explore sub nodes of current node
            elseif strcmpi(filtering_rule, 'BFS')
                mode_search = 1;
            end
            % Ascending or descending search? (from an empty message we go up and construct a clique one fanal at a time, or we descend from the full message and remove one fanal at a time until we have a clique?)
            mode_asc = 1;
            if strcmpi(filtering_rule, 'MLD') % ML Descending
                mode_asc = 0;
            end
            % Other modes settings
            no_double_visit = false; % prevent generating sub nodes which we already visited in the past?
            precompute_dead_ends = true; % precompute all possible dead ends (domain elimination) before exploring solutions? This is very memory intensive but will speed up the exploration a lot.
            % Total number of dropped messages (because they were too long to converge)
            dropped_count = 0;
            resign_count = 0;
            % Loop for each tampered message to test
            for i=1:mpartial
                if ~silent; printf('ML message %i/%i\n', i, mpartial); aux.flushout(); end;
                msg = logical(propag(:,i)); % Extract the message
                found_flag = false; % Flag is true if we have found a k-clique (or if concurrent_cliques > 1, until we find at least concurrent_cliques k-cliques)
                resign_flag = false; % Flag is true if we have expanded all nodes (open is empty) and we still haven't found any k-clique
                just_found = false;
                % Check that the message has at least enough nodes to form a k-clique of order c, because else that means we can't even possibly find a clique of order c for this message. We just skip it
                if nnz(msg) < c % The message does not have enough nodes to form a clique
                    out(:,i) = logical(msg); % If there's not enough nodes to form a k-clique of order c, then we keep this message as-is and skip it for the next iteration (if there's only one iteration, this message will obviously be wrong anyway, but with multiple iterations it may converge)
                    if ~silent; printf('ML not enough k-cliques found, resign!\n'); aux.flushout(); end;
                    resign_count = resign_count + 1;
                else % Else the message has enough nodes to form a clique
                    % Init some vars
                    if no_double_visit; closed = sparse([]); end; % List of already visited nodes (so that we won't visit the same node twice)
                    dead_ends = sparse([]); % List of nodes that won't lead to a clique by using domain-elimination (if a combination of node A-B-C does not form a clique, then we avoid exploring this subtree entirely, meaning that we will avoid A-B-C-D, A-B-C-E, A-B-C-D-E, etc.).
                    kcliques = sparse([]); % List of found k-cliques (we must maintain a list if we try to find several concurrent_cliques. With only one clique, it's useless, but with more than one, we must find each clique separately, and then at the end, we concatenate all the cliques together to form the final message we return)

                    % Extract all the links submatrix (= the adjacency matrix for this message)
                    pidxs = find(msg);
                    propag_links = net(pidxs, pidxs);

                    % Special case: this message is already a clique, we can keep it as-is and continue onto the next message
                    % Note: if concurrent_cliques > 1, this means that the two cliques are totally overlapping here. So we need to process it in a special case because else we will try to find two different cliques, which is not possible since they are totally overlapping.
                    if nnz(propag_links) == numel(propag_links)
                        out(:,i) = logical(msg);
                        continue;
                    end

                    % Extract the list of separate nodes (this will generate as many submessages as there are activated fanals, so that each submessage contains only one fanal)
                    activated_fanals = sparse(find(msg), 1:nnz(msg), 1, n, nnz(msg));

                    % Precompute dead ends at the beginning by computing all combinations of two activated fanals with no link between
                    % Note: this must be done before resorting activated_fanals
                    % NOTE2: this is VERY memory consuming, thus you may need to disable it on big datasets
                    if mode_asc && precompute_dead_ends
                        propag_nolinks = ~propag_links; % generate the list of missing links
                        propag_nolinks(find(triu(propag_nolinks, 0))) = 0; % symmetry trick in clique network: since it's a clique network (and not a tournament), it's symmetrical, thus we are sure to duplicate the number of dead ends (because if fanal A is not connected to fanal B, then also B isn't connected to A). To avoid that, we preprocess the adjacency submatrix by removing the upper part of the triangle (thus A will be connected to B but B won't be connected to A), thus we generate only one dead end AB and not BA. Note: we also remove all diagonals since it's totally useless to generate dead end AA.
                        repeat_nbs = sum(propag_nolinks); % get the number of missing links per fanal
                        if any(repeat_nbs) % if there's any missing link, we continue, else it's already a clique!
                            idxs = aux.rl_decode(repeat_nbs, 1:size(activated_fanals,2)); % pre-generate the number of dead ends (we here generate the indexes of the first fanal)
                            idxs_2nd_fanal = nonzeros(bsxfun(@times, pidxs, double(propag_nolinks)))'; % now generate the indexes of the second fanal to which the first fanals aren't linked to
                            idxs_2nd_fanal = idxs_2nd_fanal + [0:n:(n*(numel(idxs)-1))]; % offset the indexes to easily apply them in matlab style
                            dead_ends = activated_fanals(:, idxs); % generate the dead ends and fill them with the first fanals (repeated as many as there are missing links)
                            dead_ends(idxs_2nd_fanal) = 1; % activate the second fanals to which there is a missing link
                        end
                        clear propag_nolinks;
                    end

                    % Pre-sort fanals to better explore the tree of solutions
                    if mode_asc
                        [~, sorted_idxs] = sort(sum(propag_links), 'descend'); % Pre-sort fanals to explore first depending on number of total active links (highest number of links first)
                        activated_fanals = activated_fanals(:, sorted_idxs); % this pre-sorting will be used everytime we expand sub-nodes to explore first the nodes with highest scores
                        open = activated_fanals; % and we use this pre-sorting at the start
                    %open = activated_fanals(:,randperm(nnz(msg))); % open contains the list of nodes to explore next. At each iteration, we will pull the first node in the list. To init, we use the list of separate nodes in a random permutation
                    else
                        [~, sorted_idxs] = sort(sum(propag_links), 'ascend'); % Pre-sort fanals to explore first depending on number of total active links (lowest number of links first)
                        activated_fanals = activated_fanals(:, sorted_idxs); % this pre-sorting will be used everytime we expand sub-nodes to explore first the nodes with highest scores
                        open = msg;
                    end

                    % Main loop: try to find a k-clique until we have found one or we explored the whole tree and couldn't find any k-clique
                    counter = 0;
                    countermax = 500;
                    counter2 = 0;
                    counterlimit = 1E4;
                    while ~found_flag && ~resign_flag
                        counter = counter + 1;
                        counter2 = counter2 + 1;
                        % if this is too slow to converge, we randomize things up
                        if counter > countermax
                            if ~silent; printf('Reshuffling stack...\n', i, mpartial); aux.flushout(); end;
                            open = open(:,randperm(size(open,2)));
                            activated_fanals = activated_fanals(:,randperm(size(activated_fanals,2)));
                            counter = 0;
                            countermax = countermax * 2; % not too many shuffles, lengthen the number of iterations required for a shuffle
                            %keyboard; % sum(open) to check the evolution of the stack (this will give the score of each node)
                        end
                        if counter2 > counterlimit
                            if ~silent; printf('ML dropping message %i: too many iterations...\n', i); aux.flushout(); end;
                            dropped_count = dropped_count + 1;
                            break;
                        end
                        % If we just found a clique, and concurrent_cliques > 1 (so we are looking for another clique), we restart from the beginning to avoid exploring all siblings of current kclique which is highly unlikely to give another kclique (they are probably very different, ie they do not share many nodes, unless density is very very high)
                        if just_found == true
                            % First we resort by putting last the fanals that are in the found kcliques
                            matched_fanals = any((activated_fanals' * kcliques)', 1);
                            if mode_asc
                                activated_fanals = activated_fanals(:, [find(~matched_fanals) find(matched_fanals)]);
                                % Then reinit open list with the fanals reordered
                                open = activated_fanals;
                            else
                                activated_fanals = activated_fanals(:, [find(matched_fanals) find(~matched_fanals)]);
                                % Then reinit open list with the full message
                                open = msg;
                            end
                            just_found = false;
                        end
                        % Nothing remaining to explore? Then stop, there's no clique
                        if isempty(open)
                            if ~silent; printf('ML not enough k-cliques found, resign!\n'); aux.flushout(); end;
                            resign_count = resign_count + 1;
                            resign_flag = true;
                            break; % KO stopping criterion: we couldn't find enough (=concurrent_cliques) cliques with c fanals, we just stop here.
                        % Else we still have nodes to explore, so we proceed onto the exploration
                        else
                            cur_node = open(:,1); % Pop the first submessage to explore
                            open(:,1) = []; % And remove it from the open list
                            if no_double_visit; closed = [closed, cur_node]; end; % Add this node to the list of visited nodes

                            if (mode_asc && nnz(cur_node) <= c) || (~mode_asc && nnz(cur_node) >= c) % if number of nodes is above c then we've got nothing to do because we have more nodes than required for a clique, so this is surely not a good track to follow
                                % Get the links
                                pidxs = find(cur_node); % extract links by network matrix indexing (the crossover between indexes in column and rows will give us the links for this list of nodes)
                                propag_links = net(pidxs, pidxs); % Get the sub-matrix of all links between all nodes in our message
                                % If this isn't a clique, then it's a dead end, we add it to the list of paths that should never be expanded (any specialization of this message - meaning any message containing at least the same nodes as this message, but it can contain more - will be discarded as a dead end)
                                if ( mode_asc && ~(nnz(propag_links) == numel(propag_links)) ) % to check that this is a clique, we just check that the number of activated connections (= number of non-zeros entries) is equal to the total number of possible connections in this sub-matrix (= total number of elements in this submatrix)
                                    dead_ends = [dead_ends cur_node];
                                % Else this is a clique (all fanals are interconnected, there's no 0 anywhere in the links submatrix), then we will proceed on by either expanding the sub nodes or by finishing if this clique has c nodes
                                else
                                    % If the number of fanals in this clique is below c, then we need to explore further and add more fanals, so we can explore the sub-nodes of the current node
                                    % This is the main subsection, this is where we generate subnodes to explore and smartly filter useless subnodes and thus guide the search to more efficiently explore
                                    if (mode_asc && nnz(cur_node) < c) || (~mode_asc && nnz(cur_node) > c)
                                        sub_nodes = sparse([]);
                                        if mode_asc
                                            % sub_nodes = all combinations of current message with base characters
                                            sub_nodes = or(repmat(cur_node, 1, size(activated_fanals, 2)), activated_fanals);
                                            % remove sub nodes that are the same as the current node (they don't have one more fanal)
                                            sub_nodes = sub_nodes(:, sum(sub_nodes) > sum(cur_node)); % since we are adding one fanal at a time in each submessage, if one or several submessages have the same size as the current message, then it means that the fanal we added overlaps a fanal that was already active in the current message, thus we can without a doubt discard it as a duplicate
                                            % remove sub messages where we added a fanal in the same cluster as another already activated fanal in this sub message (ie: it's useless to explore this sub node because a clique cannot have two fanals in the same cluster, thus we are sure this will be a dead end)
                                            sub_nodes = sub_nodes(:, ~any(reshape(sum(reshape(sub_nodes, l, []), 1) > 1, Chi, []), 1));
                                        else
                                            % First try to guide which fanal we will eliminate: we will choose preferably the fanals that are the less connected, because probably these fanals aren't in the clique
                                            [~, sorted_idxs] = sort(sum(propag_links), 'ascend'); % Pre-sort fanals to explore first depending on number of total active links (lowest number of links first)
                                            [~, ~, idxs] = intersect(pidxs, find(msg)); % Find the ids of the fanals we already removed in the past (readjust the indexes offsets)
                                            activated_fanals2 = activated_fanals(:, idxs(sorted_idxs)); % delete useless subnodes and reorder to explore (remove) first the fanals that are the less connected
                                            % sub_nodes = all combinations of the difference between current message with base characters
                                            sub_nodes = repmat(cur_node, 1, size(activated_fanals2, 2));
                                            sub_nodes(find(activated_fanals2)) = 0;
                                            % remove sub nodes that are the same as the current node (they don't have one less fanal than current message)
                                            sub_nodes = sub_nodes(:, sum(sub_nodes) < sum(cur_node)); % since we are adding one fanal at a time in each submessage, if one or several submessages have the same size as the current message, then it means that the fanal we added overlaps a fanal that was already active in the current message, thus we can without a doubt discard it as a duplicate
                                        end
                                        % Open list no duplication: do not add subnodes that are already in the open list
                                        open_nodes_dup = find(any(bsxfun(@ge, (sub_nodes' * open)', sum(sub_nodes)), 2)); % find all sub_nodes that match the nodes in the open list
                                        if ~isempty(open_nodes_dup); open(:, open_nodes_dup) = []; end; % remove them (if any)
                                        clear open_nodes_dup; % clear some memory
                                        % No double visit of the same nodes
                                        % Note that it can cost a lot of memory, and cause a memory overflow on Octave since we have to remember all nodes ever visited!
                                        if no_double_visit
                                            interesting_sub_nodes = ~any(bsxfun(@ge, (sub_nodes' * closed)', sum(sub_nodes)), 1); % Compare each sub node with the list of previously visited nodes (inside the closed list)
                                            if isempty(interesting_sub_nodes) || ~any(interesting_sub_nodes) % All sub messages were deleted? So we don't have any sub node to happen, let's walk another branch of the tree
                                                sub_nodes = [];
                                            else % Else we have some interesting sub nodes to extract
                                                sub_nodes = sub_nodes(:, interesting_sub_nodes);
                                            end
                                        end
                                        % Domain filtering
                                        if mode_asc
                                            interesting_sub_nodes = ones(1,size(sub_nodes,2)); % Reinit indexes
                                            if ~isempty(dead_ends) && ~isempty(sub_nodes)
                                                interesting_sub_nodes = ~any(bsxfun(@ge, (sub_nodes' * dead_ends)', sum(dead_ends)'), 1); % Compare each sub node and see if a dead end match with these sub nodes (ie: if all the fanals of a dead end are contained in one of those sub messages, it is discarded, even if it contains more fanals that the dead end, this is because a message with c fanals will always be more general than another message with c+x fanals if they both have the same c fanals)
                                            end
                                            if isempty(interesting_sub_nodes) || ~any(interesting_sub_nodes) % All sub messages were deleted? So we don't have any sub node to happen, let's walk another branch of the tree
                                                sub_nodes = [];
                                            else % Else we have some interesting sub nodes to extract
                                                sub_nodes = sub_nodes(:, interesting_sub_nodes);
                                            end
                                        end
                                        % Append new nodes to explore if any
                                        if ~isempty(sub_nodes)
                                            % Depth-First Search: we first explore the sub nodes of the current node then later we will explore sibling nodes of current node
                                            if mode_search == 0
                                                open = [sub_nodes, open];
                                            % Breadth-First Search: we first explore sibling nodes (nodes at the same level as current node) and then after we will explore sub nodes of current node
                                            elseif mode_search == 1
                                                open = [open, sub_nodes];
                                            end
                                        end
                                    % Else if number of nodes equal to c: we just found a clique!
                                    elseif nnz(cur_node) == c && (mode_asc || ~(nnz(propag_links) == numel(propag_links)) )
                                        if concurrent_cliques == 1 || ( isempty(kcliques) || ~any(bsxfun(@eq, (cur_node' * kcliques), sum(kcliques))) ) % Important: Check that the k-clique we just found is not a duplicate of a k-clique we found previously (only useful if we have to find several k-cliques, ie: when concurrent_cliques > 1)
                                            kcliques = [kcliques, cur_node]; % add the current sub message into the list of found cliques
                                            if mode_asc; dead_ends = [dead_ends, cur_node]; end; % avoid re-exploring this same solution twice
                                            just_found = true;
                                            if ~silent; printf('Clique %i found!\n', size(kcliques,2)); aux.flushout(); end;
                                        end
                                        % If we have reached the number of cliques to find (= concurrent_cliques), then we can construct the out message and stop here
                                        if size(kcliques, 2) == concurrent_cliques % we can't know if the concurrent_cliques have to overlap or not, so as long as we find 2 different cliques, we take it as a result (we are guaranteed to find different cliques everytime because we add all found kcliques in the dead_ends just above)
                                            out(:,i) = any(kcliques, 2); % concatenate all found cliques into one single message
                                            found_flag = true;
                                            break; % OK stopping criterion: found the cliques, they may be the wrong ones if multiple cliques are available and thus there's some kind of confusion, but at least we've found something!
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

                %if found_flag
                %    out(:,i) = any(kcliques, 2);
                %end % else do nothing, since out is already zero'ed, this means that the current message will be all 0's meaning we didn't find a solution (clique)
            end

            if ~silent
                printf('ML total number of dropped messages: %i/%i\n', dropped_count, mpartial);
                printf('ML total number of resigned messages: %i/%i\n', resign_count, mpartial);
                aux.flushout();
            end

        % Else error, the filtering_rule does not exist
        else
            error('Unrecognized filtering_rule: %s', filtering_rule);
        end
        % TODO: add GLsKO (kick all losers with kth minimum score) and oGLsKO (kick all losers with global minimum score)

        
        % 4- Some post-processing

        % Disambiguation by overlays majority voting
        % Note: the first two methods are deprecated because they give lower performance boost than the faithful Ehsan version (the third method, in the else clause)
        if enable_overlays
            % Overlays filtering: instead of filtering at propagation time, we first propagate, then filter using GWsTA just like usually, and then only we filter by tags, just like a guiding mask but totally unsupervised
            if strcmpi(propagation_rule, 'overlays_old_filter')
                fastmode = @(x) max(aux.fastmode(nonzeros(x))); % prepare the fastmode function (in case of draw between two values which are both the mode, keep the maximum one = most recent one)
                winner_overlays = gmtimes(double(out'), net, fastmode, @times, [mpartial, 1])'; % compute the mode-of-products to fetch the major tag per message
                out = out .* bsxfun(@times, double(out), winner_overlays); % prepare the input messages with the mode (instead of binary message, each message will be assigned the id of its major tag)
                out = gmtimes(out', net, @sum, @(a,b) and(full(a>0), bsxfun(@eq, a, b))); % finally, compute the sum-of-equalities: we activate only the edges corresponding to the major tag for this message.
                %@(a,b) and(full(a > 0), eq(full(a),full(b)))
                out = out';
            % Overlays filtering ala Ehsan - WRONG this is a variation of what he is doing
            elseif strcmpi(propagation_rule, 'overlays_old_filter2') && l > 1
                %ambiguity_mask = bsxfun(@times, out2, sum(out) >= c); % filter out unambiguous messages (having c fanals activated)
                % filter out unambiguous clusters (where only one fanal is activated) - we will only use tags to disambiguate ambiguous clusters, for the others we can keep the result of the sum-of-sum propagation
                out = reshape(out, l, mpartial * Chi);
                %out(1) = 1; out(2) = 1; % debug
                ambiguity_mask = repmat(sum(out, 1) >= 2, l, 1); % filter out unambiguous clusters (keep only ambiguous clusters)
                out = bsxfun(@and, out, sum(out, 1) == 1); % btw filter out ambiguous clusters from the final messages, since we will later repush the disambiguated clusters
                out = reshape(out, n, mpartial);
                ambiguity_mask = reshape(ambiguity_mask, n, mpartial);

                if nnz(ambiguity_mask) > 0
                    % DISAMBIGUATION BY TAGS step
                    nnzcolmode = @(x) aux.nnzcolmode(x); % prepare the fastmode function (in case of draw between two values which are both the mode, keep the maximum one = most recent one)
                    % Find the major tag (mode) of incoming edges for each ambiguous fanal
                    %gmtimes(double(partial_messages), bsxfun(@times, ambiguity_mask(:,i)', net), fastmode, @times);
                    out2 = sparse(n, mpartial);
                    parfor i=1:mpartial
                        if nnz(ambiguity_mask(:,i)) > 0
                            %tt = bsxfun( @times, double(partial_messages(:,1)), bsxfun(@times, double(ambiguity_mask(:,1)'), net) ); tt(1) = 4; tt(2) = 4; % debug
                            [~, argmode] = nnzcolmode( bsxfun( @times, double(partial_messages(:,i)), bsxfun(@times, double(ambiguity_mask(:,i)'), net) ) );
                            argmode(isnan(argmode)) = 0;
                            [I, J, V] = find(argmode);
                            out2(:,i) = sparse(I, J, V, n, 1);
                        end
                    end

                    % Find the fanal with highest number of major tags (the winner of the disambiguation)
                    out2 = reshape(out2, l, mpartial * Chi);
                    winner_overlays = max(out2);
                    nnzidxs = find(winner_overlays); % to avoid finding winners where there in fact are only zeros values
                    out2(:, nnzidxs) = bsxfun(@eq, out2(:, nnzidxs), winner_overlays(nnzidxs)); % Note: there can still be an ambiguity if two fanals have equally the same number of edges with the major tag! (and the tag may be different for both!)
                    out2 = reshape(out2, n, mpartial);

                    % Finally, push back the disambiguated clusters
                    out = or(out, out2);
                end
            % Overlays filtering ala Ehsan, correct and faithful version
            % this overlays filtering must only be applied after propagation + filtering. This is a post-processing step to remove ambiguity.
            else
                for mi = 1:mpartial
                    % Pick one message
                    decoded_fanals = find(out(:, mi));

                    % Filter edges going outside the clique
                    decoded_edges = net(decoded_fanals, decoded_fanals) ; % fetch tags of only the edges of the recovered clique (thus edges connected between fanals in the recovered clique but not those that are not part of the clique, ie connected to another fanal outside the clique, won't be included)
                    if nnz(decoded_edges) == 0; continue; end; % if this message is empty then just quit

                    % Filter useless fanals (fanals that do not possess as many edges as the maximum - meaning they're not part of the clique). Note: This is a pre-processing enhancement step, but it's not necessary if you use fastmode(nonzeros(decoded_edges)) (the nonzeros will take care of the false 0 tag) BUT it greatly enhances the performances when using iterations > 1.
                    if concurrent_cliques == 1 && ~concurrent_disequilibrium % use GWTA to filter if we have only one clique and no disequilibrium, because if we have multiple concurrent cliques this won't work because the cliques may have different number of edges and shared fanals may have a lot of edges, but only this shared fanal will be kept and the others correct fanals will be filtered out because they have less edges than the shared fanal. Also avoid if concurrent_disequilibrium is enabled, for similar reasons AND because we don't want to filter out possibly correct fanals, which this does (check with matching measure, this trick here lowers down the matching).
                        % Filter using GWTA (keep only the fanals with max score)
                        fanal_scores = sum(sign(decoded_edges));
                        winning_score = max(fanal_scores);
                        gwta_mask = (fanal_scores == winning_score);
                        decoded_edges = decoded_edges(gwta_mask, :) ;
                    else % else i we have concurrent cliques and/or disequilibrium, use GWSTA to filter (because GWTA won't work in concurrent case).
                    % NOTE: this is even less necessary than in the non concurrent cliques case, here it only provides a small performance boost (but significative), so it's up to you to see if the additional CPU time needed to do the GWSTA filtering is worth it, but keep in mind that filtering also speeds up the tags filtering below, because we remove fanals whose edges won't have to be tag checked!
                        fanal_scores = sum(sign(decoded_edges));
                        if numel(fanal_scores) > (c * concurrent_cliques)
                            % Filter using GWSTA (reject fanals below the kth max score)
                            max_scores = sort(fanal_scores,'descend'); % sort scores
                            kmax_score = max_scores(k); % get the kth max score
                            kmax_score(kmax_score == 0) = realmin(); % No false winner trick: filter out kth winners that have value 0 by replacing the threshold with the minimum real value (greater than 0)
                            gwsta_mask = fanal_scores >= kmax_score; % filter out all fanals that get a score below the kth winner
                            decoded_edges = decoded_edges(gwsta_mask, :); % update our network
                            
                            % Alternative methods with lesser performances
                            %decoded_edges = decoded_edges((fanal_scores ~= min(nonzeros(fanal_scores))), :);
                            %decoded_edges = decoded_edges(ismember(fanal_scores, aux.fastmode(nonzeros(fanal_scores))), :);
                        end
                    end

                    % Filter edges having a tag different than the major tag, and then filter out fanals that gets disconnected from the clique (all their incoming edges were filtered because they were of a different tag than the major tag)
                    if concurrent_cliques == 1
                        major_tag = aux.fastmode(nonzeros(decoded_edges)) ; % get the major tag (the one which globally appears the most often in this clique). NOTE: nonzeros somewhat slows down the processing BUT it's necessary to ensure that 0 is not chosen as the major tag (since it represents the absence of edge!) - this problem often happens when using a sparse network (Chi > c).
                        decoded_edges(decoded_edges ~= min(major_tag)) = 0 ; % shutdown edges who haven't got the maximum tag. NOTE: in case of ambiguity (two or more major tags), we keep the minimum (oldest) one.
                    else
                        major_tag = aux.kfastmode(nonzeros(decoded_edges), concurrent_cliques);
                        decoded_edges(~ismember(decoded_edges, major_tag)) = 0; % shutdown edges who haven't got the maximum tag. NOTE: in case of ambiguity (two or more major tags), we keep them all. This seems to enhance performances a bit compared to select the minimum one or a random one.
                    end
                    decoded_fanals = decoded_fanals(sum(decoded_edges) ~= 0) ; % kick out fanals which have no incoming edges after having deleted edges without major tag (ie: nodes that become isolated because their edges had different tags than the major tag will just be removed, because if these nodes become isolated it's because they obviously are part of another message, else they would have at least one edge with the correct tag).

                    % Finally, replace the disambiguated message back into the stack
                    out(:,mi) = sparse(decoded_fanals, 1, 1, n, 1);
                end
            end
        end

        % Residual memory
        if residual_memory > 0
            out = out + (residual_memory .* partial_messages); % residual memory: previously activated nodes lingers a bit (a fraction of their activation score persists) and participate in the next iteration
        end

        % Auxiliary support network post-processing (tests)
        if isfield(cnetwork, 'auxiliary') && strcmpi(cnetwork_choose, 'primary')

            ttmode = 3;

            if ttmode == 0 % useless
                propag2 = propag + mes_echo;
                
                % GWsTA
                max_scores = sort(propag2,'descend');
                kmax_score = max_scores(k, :);
                kmax_score(kmax_score == 0) = realmin(); % No false winner trick: better version of the no false winner trick because we replace 0 winner scores by the minimum real value, thus after we will still get a sparse matrix (else if we do the trick only after the bsxfun, we will get a matrix filled by 1 where there are 0 and the winner score is 0, which is a lot of memory used for nothing).
                out2 = logical(bsxfun(@ge, propag2, kmax_score));
                if aux.isOctave(); out = sparse(out2); end; % Octave's bsxfun breaks the sparsity...
                out2 = and(propag2, out2); % No false winner trick: avoids that if the kth max score is in fact 0, we choose 0 as activating score (0 scoring nodes will be activated, which is completely wrong!). Here we check against the original propagation matrix: if the node wasn't activated then, it shouldn't be now after filtering.
                out = out2;

            elseif ttmode == 1 % better
                % excitation + GWSTA
                propag2 = double(out);
                propag2(find(out)) = (propag+mes_echo)(find(out)); % propag+logical(mes_echo) ?

                % GWsTA
                max_scores = sort(propag2,'descend');
                kmax_score = max_scores(k, :);
                kmax_score(kmax_score == 0) = realmin(); % No false winner trick: better version of the no false winner trick because we replace 0 winner scores by the minimum real value, thus after we will still get a sparse matrix (else if we do the trick only after the bsxfun, we will get a matrix filled by 1 where there are 0 and the winner score is 0, which is a lot of memory used for nothing).
                out2 = logical(bsxfun(@ge, propag2, kmax_score));
                if aux.isOctave(); out = sparse(out2); end; % Octave's bsxfun breaks the sparsity...
                out2 = and(propag2, out2); % No false winner trick: avoids that if the kth max score is in fact 0, we choose 0 as activating score (0 scoring nodes will be activated, which is completely wrong!). Here we check against the original propagation matrix: if the node wasn't activated then, it shouldn't be now after filtering.
                out = out2;

            elseif ttmode == 2 % useless
                % Inhibition
                out(~mes_echo) = 0;

            elseif ttmode == 3 % better
                % inhibition + GWSTA)
                propag2 = double(out);
                propag2(find(out)) = (propag - ~mes_echo)(find(out));
                propag2(propag2 < 0) = 0;

                % GWsTA
                max_scores = sort(propag2,'descend');
                kmax_score = max_scores(k, :);
                kmax_score(kmax_score == 0) = realmin(); % No false winner trick: better version of the no false winner trick because we replace 0 winner scores by the minimum real value, thus after we will still get a sparse matrix (else if we do the trick only after the bsxfun, we will get a matrix filled by 1 where there are 0 and the winner score is 0, which is a lot of memory used for nothing).
                out2 = logical(bsxfun(@ge, propag2, kmax_score));
                if aux.isOctave(); out = sparse(out2); end; % Octave's bsxfun breaks the sparsity...
                out2 = and(propag2, out2); % No false winner trick: avoids that if the kth max score is in fact 0, we choose 0 as activating score (0 scoring nodes will be activated, which is completely wrong!). Here we check against the original propagation matrix: if the node wasn't activated then, it shouldn't be now after filtering.
                out = out2;
            % Either GLsKO either oGLKO or GWSTA or inhibition (tous ceux qui ne sont pas liés au réseau aux sont éteints)
            end

        end

        partial_messages = out; % set next messages state as current
        if ~silent; aux.printtime(toc()); end;
    end

    % Disequilibrium post-processing: we remove the clique we just found from the messages
    if concurrent_cliques_bak > 1 && concurrent_disequilibrium
        %bsxfun(@and, sum(out_final) < (concurrent_cliques * c), out) % avoid adding more fanals if we already found enough fanals to cover all cliques
        out_final = or(out_final, out);
        if diter < diterations
            partial_messages = partial_messages_bak; % reload the full messages
            partial_messages(find(out_final)) = 0; % remove all cliques we have found up until now to focus only on cliques we didn't find yet. NOTE: be careful not to use find(out) instead of find(out_final), because out contains only the latest clique found, not all the previous cliques, and this will lower down performances a lot!
            %a = aux.interleaven(2, partial_messages_bak, out, partial_messages); full([sum(a); a])
        end
        % TODO: adapt guiding mask. NO: we can't adapt the guiding mask because we don't know where to look at.
        % TODO: with concurrent_disequilibrium == 3, try to find when there are errors: when both cliques are found at the first iteration? If that's the case, try to redo the iteration only for these messages by using another random fanal
    end
end

% -- After-convergence post-processing
% Disequilibrium final post-processing: we set the final messages to be returned (= the concatenation of all cliques found separately via disequilibrium)
if concurrent_cliques_bak > 1 && concurrent_disequilibrium
    partial_messages = out_final;
end

if residual_memory > 0 % if residual memory is enabled, we need to make sure that values are binary at the end, not just near-binary (eg: values of nodes with 0.000001 instead of just 0 due to the memory addition), else the error can't be computed correctly since some activated nodes will still linger after the last WTA!
    partial_messages = max(round(partial_messages), 0);
end

% At the end, partial_messages contains the recovered, corrected messages (which may still contain errors, but the goal is to have a perfect recovery!)
% The performance, error test must be done by yourself, by comparing the returned partial_messages with your original messages.

end % endfunction
