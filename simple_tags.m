% Simple usage of the tagged network (an extension of the cliques network using coloured edges, so that we can disambiguate in case of spurious fanals)

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

% Primary network params
m = 0.7; %1E4;
miterator = 0;
c = 8; % 8; % with tags, it seems that higher c is, the lower the error rate will be
l = 16; % 32;
Chi = 32; % 64;
erasures = floor(c/2);
gamma_memory = 0; % gamma 0 is best for tags
iterations = 2; % IMPORTANT: try with 1 and with 4 iterations, because both will give different results for different overlays_rule
tests = 1;
tampered_messages_per_test = 200;
filtering_rule = 'GWsTA';
propagation_rule = 'sum';
enable_guiding = false;

% Overlays / Tags
enable_overlays = true; % enable tags/overlays disambiguation?
overlays_max = 0; % 0 for maximum number of tags (as many tags as messages/cliques) ; 1 to use only one tag (equivalent to standard network without tags) ; n > 1 for any definite number of tags
overlays_interpolation = 'uniform'; % interpolation method to reduce the number of tags when overlays_max > 1: uniform, mod or norm
enable_overlays_guiding = false; % provide the major tag without voting?

% Concurrency params
concurrent_cliques = 1;
no_concurrent_overlap = false;

% Verbose?
silent = false;

% == Launching the runs
tperf = cputime();
[cnetwork, thriftymessages, density] = gbnn_learn('m', m, 'l', l, 'c', c, 'Chi', Chi, 'miterator', miterator, 'enable_overlays', enable_overlays, 'silent', silent);

if ~silent
    fprintf('Minimum overlay: %i\n',  full(min(cnetwork.primary.net(cnetwork.primary.net > 0))));
end

error_rate = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                  'iterations', iterations, ...
                                                                                  'tests', tests, 'tampered_messages_per_test', tampered_messages_per_test, ...
                                                                                  'enable_guiding', enable_guiding, 'filtering_rule', filtering_rule, 'propagation_rule', propagation_rule, 'erasures', erasures, 'gamma_memory', gamma_memory, ...
                                                                                  'concurrent_cliques', concurrent_cliques, 'no_concurrent_overlap', no_concurrent_overlap, ...
                                                                                  'enable_overlays', enable_overlays, 'overlays_max', overlays_max, 'overlays_interpolation', overlays_interpolation, 'enable_overlays_guiding', enable_overlays_guiding, ...
                                                                                  'silent', silent);

if ~silent
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed
end

% The end!
