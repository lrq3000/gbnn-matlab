% Tags + Disequilibrium applied to concurrent cliques problem. Works great!

% Clear things up
clear all; % don't forget to clear all; before, else some variables or sourcecode change may not be refreshed and the code you will run is the one from the cache, not the latest edition you did!
close all;

% Importing auxiliary functions
aux = gbnn_aux; % works with both MatLab and Octave

% Primary network params
m = 40E2;
c = 8;
l = 16;
Chi = 32;
erasures = 3;

tampered_messages_per_test = 100;
tests = 1;

iterations = 4;
gamma_memory = 1;
propagation_rule = 'sum';
filtering_rule = 'GWSTA';
filtering_rule_first_iteration = false;
enable_guiding = false;

% Concurrency params
concurrent_cliques = 3;
no_concurrent_overlap = false;

% Overlays / Tags
enable_overlays = true; % enable tags/overlays disambiguation?
overlays_max = 0; % 0 for maximum number of tags (as many tags as messages/cliques) ; 1 to use only one tag (equivalent to standard network without tags) ; n > 1 for any definite number of tags
overlays_interpolation = 'uniform'; % interpolation method to reduce the number of tags when overlays_max > 1: uniform, mod or norm

% Concurrent disequilibrium trick
concurrent_disequilibrium = 3; % 1 for superscore mode, 2 for one fanal erasure, 3 for nothing at all just trying to decode one clique at a time without any trick, 0 to disable

% Verbose?
silent = false;

% == Launching the runs
tperf = cputime();
[cnetwork, thriftymessages, density] = gbnn_learn('m', m, 'l', l, 'c', c, 'Chi', Chi, 'enable_overlays', enable_overlays, 'silent', silent);

if ~silent
    printf('Minimum overlay: %i\n',  full(min(cnetwork.primary.net(cnetwork.primary.net > 0))));
end

error_rate = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                  'iterations', iterations, ...
                                                                                  'tests', tests, 'tampered_messages_per_test', tampered_messages_per_test, ...
                                                                                  'enable_guiding', enable_guiding, 'filtering_rule', filtering_rule, 'propagation_rule', propagation_rule, 'erasures', erasures, 'gamma_memory', gamma_memory, ...
                                                                                  'concurrent_cliques', concurrent_cliques, 'no_concurrent_overlap', no_concurrent_overlap, 'concurrent_disequilibrium', concurrent_disequilibrium, ...
                                                                                  'enable_overlays', enable_overlays, 'overlays_max', overlays_max, 'overlays_interpolation', overlays_interpolation, ...
                                                                                  'silent', silent);

if ~silent
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed
end

% The end!
