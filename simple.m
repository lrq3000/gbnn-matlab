% Example of the simplest call script for the gbnn network

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
m = 15E3;
c = 8;
l = 32;
Chi = 64;
erasures = floor(c*0.5);
iterations = 3;
propagation_rule = 'sum';
filtering_rule = 'GWsTA';
enable_guiding = false;
tampering_type = 'erase';

tampered_messages_per_test = 1000;

% == Launching the runs
tperf = cputime();
[cnetwork, thriftymessages, density] = gbnn_learn('m', m, 'l', l, 'c', c, 'Chi', Chi);
error_rate = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, 'erasures', erasures, 'tampering_type', tampering_type, 'iterations', iterations, 'propagation_rule', propagation_rule, 'filtering_rule', filtering_rule, 'tampered_messages_per_test', tampered_messages_per_test, 'enable_guiding', enable_guiding);
aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do everything: %g seconds.\n'); aux.flushout(); % print total time elapsed

% The end!
