% Example of the simplest call script for the gbnn network

% Clear things up
clear all; % don't forget to clear all; before, else some variables or sourcecode change may not be refreshed and the code you will run is the one from the cache, not the latest edition you did!
close all;

% Importing auxiliary functions
aux = gbnn_aux; % works with both MatLab and Octave

% Primary network params
m = 1000; % 500; % 10000;
c = 6; % 8;
l = 16; % 32;
Chi = 12; % 64;
gamma_memory = 1;
iterations = 4;
tests = 1;
tampered_messages_per_test = m;
filtering_rule = 'GWsTA';
erasures = 1; %floor(c/2);
enable_guiding = false;
enable_dropconnect = false;
dropconnect_p = 0;

% Training params (auxiliary network)
train = true;
c2 = 1;
l2 = l*400;
Chi2 = 1;
training_batchs = 1;
trainsetsize = m*training_batchs; %floor(m/trainingbatchs);
no_auxiliary_propagation = false; % false for concur cliques
train_on_full_cliques = 0; % false for concur cliques
train_enable_dropconnect = false;
train_dropconnect_p = 0.9;
train_subsampling_p = []; % [] to disable, value between 0 and 1 to enable

% Concurrency params
concurrent_cliques = 1;
no_concurrent_overlap = true;

% Verbose?
silent = false;

% == Launching the runs
tperf = cputime();
[cnetwork, thriftymessages, density] = gbnn_learn('m', m, 'l', l, 'c', c, 'Chi', Chi, 'silent', silent);

if train
    fprintf('Training the network...\n'); aux.flushout();
    [cnetwork, real_density_aux, real_density_bridge, auxfullcell] = gbnn_train('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, 'l', l2, 'c', c2, 'Chi', Chi2, ...
                                             'tampered_messages_per_test', trainsetsize, 'training_batchs', training_batchs, 'no_auxiliary_propagation', no_auxiliary_propagation, 'train_on_full_cliques', train_on_full_cliques, ...
                                             'iterations', iterations, 'enable_guiding', enable_guiding, 'gamma_memory', gamma_memory, 'filtering_rule', filtering_rule, 'erasures', erasures, ...
                                             'concurrent_cliques', concurrent_cliques, 'no_concurrent_overlap', no_concurrent_overlap, ...
                                             'enable_dropconnect', train_enable_dropconnect, 'dropconnect_p', train_dropconnect_p, 'subsampling_p', train_subsampling_p, ...
                                             'silent', true);
end

error_rate = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                  'iterations', iterations, ...
                                                                                  'tests', tests, 'tampered_messages_per_test', tampered_messages_per_test, ...
                                                                                  'enable_guiding', enable_guiding, 'filtering_rule', filtering_rule, 'erasures', erasures, 'gamma_memory', gamma_memory, ...
                                                                                  'concurrent_cliques', concurrent_cliques, 'no_concurrent_overlap', no_concurrent_overlap, ...
                                                                                  'enable_dropconnect', enable_dropconnect, 'dropconnect_p', dropconnect_p, ...
                                                                                  'silent', silent);

if ~silent
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed

    if train
        real_density_aux
        real_density_bridge
        if isfield(cnetwork, 'auxiliary')
            cliques_prim2aux_mean = cnetwork.auxiliary.args.cliques_prim2aux_mean
            cliques_aux2prim_mean = cnetwork.auxiliary.args.cliques_aux2prim_mean
        end
        
        dispc = @(x) {full(auxfullcell{1}(:,x)), full(auxfullcell{2}(:,x))}; % displays first learned messages with first auxiliary support message
        dispf = @(x){full(auxfullcell{1}), full(auxfullcell{2})};
        disp1 = [cnetwork.auxiliary.args.links_prim2aux; cnetwork.auxiliary.args.cliques_prim2aux];
        disp2 = [cnetwork.auxiliary.args.links_aux2prim; cnetwork.auxiliary.args.cliques_aux2prim];
        % simple_support; dispc(1)
    end
end

% A = (dispc(2){1}'*cnetwork.auxiliary.prim2auxnet)'
% B = (A' * cnetwork.auxiliary.prim2auxnet')'
% [dispc(2){1} B propag(:,4)]
% ESSAYER: enlever logical dans gbnn_correct.m au moment de propag dans reseau auxiliaire + essayer de GWTA (garder que les winners): A2 = bsxfun(@ge, A, max(A))
% B2 = (A2' * cnetwork.auxiliary.prim2auxnet')'
% A3 = logical(A);
% B3 = (A3' * cnetwork.auxiliary.prim2auxnet')'
% [dispc(2){1} B B2 B3]
% [dispc(2){1} I+B I+B2 I+B3]
% [A3'; full(cnetwork.auxiliary.prim2auxnet)]


% The end!
