function [cnetwork, real_density_aux, real_density_bridge, auxfullcell] = gbnn_train(varargin)
%
% [cnetwork, thriftymessages, density] = gbnn_learn(cnetwork, thriftymessagestest, ...
%                                                                       l, c, Chi, ...
%                                                                       silent)
%
% Trains a network: disambiguate conflicting memorized messages by constructing an auxiliary network everytime a message cannot be retrieved unambiguously (without spurious fanals).
%
% This function supports named arguments, use it like this:
% gbnn_train('cnetwork', mynetwork, 'thriftymessagestest', thriftymessagestest, 'l', 4, 'c', 3)
%
% NOTE: networks arguments passed here (like l, c, Chi) are the parameters for the auxiliary network, not for the primary! For example, auxiliary's c should be a lot smaller than primary's c.
%
% Parameters are the same as gbnn_test.m (they will be passed onto test).
%

% == Importing some useful functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% == Arguments processing
% List of possible arguments and their default values
arguments_defaults = struct( ...
    ... % Mandatory
    'cnetwork', [], ...
    'thriftymessagestest', [], ...
    'l', 0, ...
    'c', 0, ...
    ...
    ... % 2014 sparse enhancement
    'Chi', 0, ...
    ...
    'no_auxiliary_propagation', false, ...
    'train_on_full_cliques', false, ...
    'training_batchs', 1, ...
    ...
    'subsampling_p', [], ...
    'enable_dropconnect', false, ...
    'dropconnect_p', 0.5, ...
    ...
    ... % Debug stuffs
    'silent', false);

% Process the arguments
arguments = aux.getnargs(varargin, arguments_defaults, 2);

% Load variables into local namespace (called workspace in MatLab)
aux.varspull(arguments);

% == Sanity Checks
if isempty(cnetwork) || isempty(thriftymessagestest) || l == 0 || c == 0
    error('Missing arguments: cnetwork, thriftymessagestest, l and c are mandatory!');
end

% == Show vars (just for the record)
if ~silent
    % -- Network variables
    l
    c
    Chi % -- 2014 update
end

% -- A few error checks
if numel(c) ~= 1
    error('c contains too many values! numel(c) should be equal to 1 or 2.');
end

n = Chi*l;

if ~silent; totalperf = cputime(); end; % for total time perfs


% #### Training phase

% Combine structures (append the arguments)
%pairs = [fieldnames(cnetwork.auxiliary), struct2cell(cnetwork.auxiliary); fieldnames(arguments), struct2cell(arguments)].';
%cnetwork.auxiliary = struct(pairs{:});

varargin = aux.delarg({'l', 'c', 'Chi', 'no_auxiliary_propagation', 'train_on_full_cliques', 'training_batchs', 'enable_dropconnect', 'dropconnect_p', 'subsampling_p',}, varargin);

if nargout >= 4
    auxfullcell = {[]; []};
end

if ~silent; fprintf('#### Training phase (try to remember messages and disambiguate conflicting messages with the help of an auxiliary network)\n'); aux.flushout(); end;
for tb=1:training_batchs
    if ~train_on_full_cliques
        [~, ~, ~, error_per_message, testset] = gbnn_test(varargin);
    elseif train_on_full_cliques == 2
        testset = thriftymessagestest';
    else
        testset = thriftymessagestest';
        propag = (testset' * cnetwork.primary.net)';
        propag(propag < cnetwork.primary.args.c) = 0;
        propag = logical(propag);

        error_per_message = any(testset ~= propag);
    end

    if train_on_full_cliques ~= 2
        testset = testset(:, error_per_message);
    end
    testset = unique(testset', 'rows')';
    m = size(testset, 2);
    if m > 0
        if ~isfield(cnetwork, 'auxiliary')
            cnetwork.auxiliary = struct( ...
                'net', sparse(n, n), ...
                'prim2auxnet', sparse(cnetwork.primary.args.n, n), ...
                'args', struct( ...
                    'l', l, ...
                    'c', c, ...
                    'Chi', Chi, ...
                    'n', n, ...
                    'enable_dropconnect', enable_dropconnect, ...
                    'dropconnect_p', dropconnect_p) ...
                );
            %varargin = aux.editarg('cnetwork', cnetwork, varargin);
        end

        % sparse([1 1 2 2 3 3], [1 2 1 2 1 2], 1, 6, 2)
        J = repmat(1:c, 1, m);
        I = repmat(1:m, c, 1); I = I(:);
        vals = randi([1 l], m*c, 1);

        rand_aux_fanals = aux.shake(sparse(I, J, vals, m, Chi), 2);
        rand_aux_fanals = gbnn_messages2thrifty(rand_aux_fanals, l);

        cnetwork.auxiliary.prim2auxnet = or(cnetwork.auxiliary.prim2auxnet, gbnn_construct_network(testset', rand_aux_fanals, [], [], subsampling_p));
        if ~no_auxiliary_propagation
            cnetwork.auxiliary.net = or(cnetwork.auxiliary.net, gbnn_construct_network(rand_aux_fanals, rand_aux_fanals));
        end

        if nargout >= 4
            auxfullcell{1} = [auxfullcell{1}; testset];
            auxfullcell{2} = [auxfullcell{2}; rand_aux_fanals'];
        end

    end
end

% == Compute density (if an auxiliary network was necessary to disambiguate the network)
real_density_aux = 0;
real_density_bridge = 0;
if isfield(cnetwork, 'auxiliary')
    real_density = full(  (nnz(cnetwork.auxiliary.net) - nnz(diag(cnetwork.auxiliary.net))) / (max(Chi*(Chi-1),1) * l^2)  );
    real_density_bridge = full(  (nnz(cnetwork.auxiliary.prim2auxnet) - nnz(diag(cnetwork.auxiliary.prim2auxnet))) / (max(cnetwork.primary.args.Chi*(Chi-1), 1) * (cnetwork.primary.args.l * l))  );

    cnetwork.auxiliary.args.density = real_density_aux;
    cnetwork.auxiliary.args.density_bridge = real_density_bridge;
    cnetwork.auxiliary.args.links_prim2aux = full(sum(cnetwork.auxiliary.prim2auxnet'));
    cnetwork.auxiliary.args.links_aux2prim = full(sum(cnetwork.auxiliary.prim2auxnet));
    cnetwork.auxiliary.args.cliques_prim2aux = cnetwork.auxiliary.args.links_prim2aux / cnetwork.auxiliary.args.c;
    cnetwork.auxiliary.args.cliques_aux2prim = cnetwork.auxiliary.args.links_aux2prim / cnetwork.primary.args.c;
    cnetwork.auxiliary.args.cliques_prim2aux_mean = mean(nonzeros(cnetwork.auxiliary.args.cliques_prim2aux)); % mean auxiliary clique per primary fanal
    cnetwork.auxiliary.args.cliques_aux2prim_mean = mean(nonzeros(cnetwork.auxiliary.args.cliques_aux2prim)); % mean number of primary clique linked to one auxiliary fanal
    
end

if ~silent
    fprintf('-- Computing density\n'); aux.flushout();
    real_density_aux
    real_density_bridge
end

if ~silent; aux.printcputime(cputime - totalperf, 'Total elapsed cpu time for training is %g seconds.\n'); aux.flushout(); end;

if ~silent; fprintf('=> Training done!\n'); aux.flushout(); end;

end % endfunction
