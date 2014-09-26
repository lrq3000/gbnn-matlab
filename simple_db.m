% Simple database search example to decode (indexed memory)
% There are two ways to use the database:
% 1- after cliques network decoding, to check if the decoded messages, even with an error, are closer to the correct answer or not. (since dbsearch returns the most similar results).
% 2- alone instead of cliques network, this will then be a simple (indexed) database search by similarity.

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

tampered_messages_per_test = 1000;
concurrent_cliques = 2;
no_concurrent_overlap = false;
concurrent_disequilibrium = false;

iterations = 1;
filtering_rule = 'GWSTA-ML'; % you should use GWSTA-ML for a database search

silent = true;


% == Launching the runs
tperf = cputime();

% Generate a database of random messages and learn a cliques network
[cnetwork, learned_messages, density] = gbnn_learn('m', m, 'l', l, 'c', c, 'Chi', Chi, 'silent', silent);
% Generate a sample of messages to test and decode using cliques network
[error_rate_cliques, ~, ~, ~, test_messages, decoded_messages] = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', learned_messages, 'tampered_messages_per_test', tampered_messages_per_test, 'filtering_rule', filtering_rule, 'concurrent_cliques', concurrent_cliques, 'iterations', iterations, 'erasures', erasures, 'no_concurrent_overlap', no_concurrent_overlap, 'concurrent_disequilibrium', concurrent_disequilibrium, 'silent', silent);

% 1- cliques-net + dbsearch
% Decode using cliques-net + dbsearch (indexed memory) to complement the cliques network decoding
decoded_messages = gbnn_dbsearch('learned_messages', learned_messages, 'decoded_messages', decoded_messages, 'concurrent_cliques', concurrent_cliques);
% Compute error rate cliques-net + dbsearch, to see if we do better
error_rate_cliques_and_dbsearch = nnz(sum((test_messages ~= decoded_messages), 1)) / size(test_messages, 2);

% 2- dbsearch only
% Decode using dbsearch solely (indexed memory, we will return the most similar learnt messages compared to the partially erased test messages)
decoded_messages = gbnn_dbsearch('learned_messages', learned_messages, 'decoded_messages', test_messages, 'concurrent_cliques', concurrent_cliques);
% Compute error rate of dbsearch
error_rate_dbsearch = nnz(sum((test_messages ~= decoded_messages), 1)) / size(test_messages, 2);

% Show error rate
error_rate_cliques
error_rate_cliques_and_dbsearch
error_rate_dbsearch

aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed

% The end!
