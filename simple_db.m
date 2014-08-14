% Simple database search example to decode

% Clear things up
clear all; % don't forget to clear all; before, else some variables or sourcecode change may not be refreshed and the code you will run is the one from the cache, not the latest edition you did!
close all;

% Importing auxiliary functions
aux = gbnn_aux; % works with both MatLab and Octave

% Vars config, tweak the stuff here
m = 19E2;
c = 8;
l = 16;
Chi = 32;
erasures = 2;

tampered_messages_per_test = 5;
concurrent_cliques = 2;
no_concurrent_overlap = false;
concurrent_disequilibrium = false;

iterations = 1;
filtering_rule = 'GWSTA-ML'; % you should use GWSTA-ML for a database search

% == Launching the runs
tperf = cputime();
[cnetwork, learned_messages, density] = gbnn_learn('m', m, 'l', l, 'c', c, 'Chi', Chi);
[~, ~, ~, ~, test_messages, decoded_messages] = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', learned_messages, 'tampered_messages_per_test', tampered_messages_per_test, 'filtering_rule', filtering_rule, 'concurrent_cliques', concurrent_cliques, 'iterations', iterations, 'erasures', erasures, 'no_concurrent_overlap', no_concurrent_overlap, 'concurrent_disequilibrium', concurrent_disequilibrium);
decoded_messages = gbnn_dbsearch('learned_messages', learned_messages, 'decoded_messages', decoded_messages, 'concurrent_cliques', concurrent_cliques);
error_rate = nnz(sum((test_messages ~= decoded_messages), 1)) / size(test_messages, 2);

error_rate

aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed

% The end!
