% Example of the simplest call script for the gbnn network

% Clear things up
clear all; % don't forget to clear all; before, else some variables or sourcecode change may not be refreshed and the code you will run is the one from the cache, not the latest edition you did!
close all;

% Importing auxiliary functions
aux = gbnn_aux; % works with both MatLab and Octave

% Vars config, tweak the stuff here
m = 10E4;
c = 8;
l = 64;
Chi = 100;

% == Launching the runs
tperf = cputime();
[network, sparsemessages, density] = gbnn_learn('m', m, 'l', l, 'c', c, 'Chi', Chi);
error_rate = gbnn_test('network', network, 'sparsemessagestest', sparsemessages, ...
                                                                                  'l', l, 'c', c, 'Chi', Chi);
aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed

% The end!
