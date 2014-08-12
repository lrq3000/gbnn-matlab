% Example of the simplest call script for the gbnn network

% Clear things up
clear all; % don't forget to clear all; before, else some variables or sourcecode change may not be refreshed and the code you will run is the one from the cache, not the latest edition you did!
close all;

% Importing auxiliary functions
aux = gbnn_aux; % works with both MatLab and Octave

% Vars config, tweak the stuff here
m = 5E2;
c = 8;
l = 16;
Chi = 32;
erasures = 3;

tampered_messages_per_test = 100;
concurrent_cliques = 2;
no_concurrent_overlap = false;

filtering_rule = 'GWSTA';
iterations = 4;
enable_guiding = false;

concurrent_disequilibrium = true;

% == Launching the runs
tperf = cputime();
[cnetwork, thriftymessages, density] = gbnn_learn('m', m, 'l', l, 'c', c, 'Chi', Chi);
error_rate = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, 'tampered_messages_per_test', tampered_messages_per_test, 'filtering_rule', filtering_rule, 'concurrent_cliques', concurrent_cliques, 'iterations', iterations, 'erasures', erasures, 'concurrent_disequilibrium', concurrent_disequilibrium, 'enable_guiding', enable_guiding, 'no_concurrent_overlap', no_concurrent_overlap);
aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed

% The end!
