% Example of the simplest call script for the gbnn network

% Clear things up
clear all; % don't forget to clear all; before, else some variables or sourcecode change may not be refreshed and the code you will run is the one from the cache, not the latest edition you did!
close all;

% Importing auxiliary functions
aux = gbnn_aux; % works with both MatLab and Octave

% Primary network params
m = 500; % 10000;
c = 8; % 8;
l = 16; % 32;
Chi = 32; % 64;
gamma_memory = 1;
iterations = 1;
tests = 1;
tampered_messages_per_test = 200;
filtering_rule = 'GWsTA';
erasures = floor(c/2);
enable_guiding = false;

% Training params (auxiliary network)
train = true;
c2 = floor(c/3);
l2 = l*2;
Chi2 = Chi*2;
training_batchs = 1;
trainsetsize = m*training_batchs; %floor(m/trainingbatchs);
no_auxiliary_propagation = true;
train_on_full_cliques = false;

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
    [cnetwork, real_density_aux, real_density_bridge] = gbnn_train('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, 'l', l2, 'c', c2, 'Chi', Chi2, ...
                                             'tampered_messages_per_test', trainsetsize, 'training_batchs', training_batchs, 'no_auxiliary_propagation', no_auxiliary_propagation, 'train_on_full_cliques', train_on_full_cliques, ...
                                             'iterations', iterations, 'enable_guiding', enable_guiding, 'gamma_memory', gamma_memory, 'filtering_rule', filtering_rule, 'erasures', erasures, ...
                                             'concurrent_cliques', concurrent_cliques, 'no_concurrent_overlap', no_concurrent_overlap, ...
                                             'silent', true);
end

error_rate = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                  'iterations', iterations, ...
                                                                                  'tests', tests, 'tampered_messages_per_test', tampered_messages_per_test, ...
                                                                                  'enable_guiding', enable_guiding, 'filtering_rule', filtering_rule, 'erasures', erasures, 'gamma_memory', gamma_memory, ...
                                                                                  'concurrent_cliques', concurrent_cliques, 'no_concurrent_overlap', no_concurrent_overlap, ...
                                                                                  'silent', silent);

if ~silent
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed

    real_density_aux
    real_density_bridge
    if isfield(cnetwork, 'auxiliary')
        mean_links_prim2aux = cnetwork.auxiliary.args.mean_links_prim2aux
        mean_links_aux2prim = cnetwork.auxiliary.args.mean_links_aux2prim
    end
end

% The end!
