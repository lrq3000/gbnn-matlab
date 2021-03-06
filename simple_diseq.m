% Simple usage of the disequilibrium trick for the concurrent cliques problem
% Works great! But works best in combination with tagged network (see simple_tags_diseq_concurrent).

% Clear things up
clear all; % don't forget to clear all; before, else some variables or sourcecode change may not be refreshed and the code you will run is the one from the cache, not the latest edition you did!
close all;

% Addpath of the whole library (this allows for modularization: we can place the core library into a separate folder)
if ~exist('gbnn_aux.m','file')
    %restoredefaultpath;
    addpath(genpath(strcat(cd(fileparts(mfilename('fullpath'))),'/gbnn-core/')));
end

% Importing auxiliary functions
aux = gbnn_aux; % works with both MatLab and Octave

% Vars config, tweak the stuff here
m = 5E2;
c = 8;
l = 16;
Chi = 32;
erasures = 3;

tampered_messages_per_test = 100;
concurrent_cliques = 3;
no_concurrent_overlap = false;

iterations = 4;
gamma_memory = 1;
filtering_rule = 'GWSTA';
filtering_rule_first_iteration = false;
enable_guiding = false;

concurrent_disequilibrium = 1; % 1 for superscore mode, 2 for one fanal erasure, 3 for nothing at all just trying to decode one clique at a time without any trick, 0 to disable

% == Launching the runs
tperf = cputime();
[cnetwork, thriftymessages, density] = gbnn_learn('m', m, 'l', l, 'c', c, 'Chi', Chi);
[error_rate, ~, ~, ~, testset] = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, 'tampered_messages_per_test', tampered_messages_per_test, 'filtering_rule', filtering_rule, 'concurrent_cliques', concurrent_cliques, 'iterations', iterations, 'erasures', erasures, 'concurrent_disequilibrium', concurrent_disequilibrium, 'enable_guiding', enable_guiding, 'no_concurrent_overlap', no_concurrent_overlap, 'filtering_rule_first_iteration', filtering_rule_first_iteration, 'gamma_memory', gamma_memory);
aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed

% The end!
