% Example of the simplest call script for the gbnn network

% Clear things up
clear all; % don't forget to clear all; before, else some variables or sourcecode change may not be refreshed and the code you will run is the one from the cache, not the latest edition you did!
close all;

% Importing auxiliary functions
aux = gbnn_aux; % works with both MatLab and Octave

% Vars config, tweak the stuff here
m = 0.5E5;
c = 12;
l = 64;
Chi = 100;
erasures = 3;

tampered_messages_per_test = 100;
concurrent_cliques = 2;

filtering_rule = 'ML';
iterations = 1;

% == Launching the runs
tperf = cputime();
[cnetwork, thriftymessages, density] = gbnn_learn('m', m, 'l', l, 'c', c, 'Chi', Chi);
error_rate = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, 'tampered_messages_per_test', tampered_messages_per_test, 'filtering_rule', filtering_rule, 'concurrent_cliques', concurrent_cliques, 'iterations', iterations, 'erasures', erasures);
aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed

% The end!
