% This example main file shows how to do a batch of runs to produce figures like those in the 2014 article.

% Clear things up
clear all;
close all;

% Addpath of the whole library (this allows for modularization: we can place the core library into a separate folder)
if ~exist('gbnn_aux.m','file')
    %restoredefaultpath;
    addpath(genpath(strcat(cd(fileparts(mfilename('fullpath'))),'/../gbnn-core/')));
end

% Importing auxiliary functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% Preparing stuff to automate the plots
% This will allow us to automatically select a different color and shape for each curve
colorvec = 'rgbkmc';
markerstylevec = '+o*.xsd^v><ph';
linestylevec = {'-' ; '--' ; ':' ; '-.'};

% Vars config, tweak the stuff here
C = [8 12 16 20]; % this is a vector because we will try several different values of c (order of cliques)
M = 1:1:10; % and different values of m (number of messages, which influences the density)
Mcoeff = 1E4;
miterator = zeros(1,numel(M)); %M/2;
l = 64;
Chi = 100;
erasures = floor(C/2);
iterations = 4;
tampered_messages_per_test = 200;
tests = 1;

enable_guiding = true;
gamma_memory = 0;
threshold = 0;
propagation_rule = 'sum';
filtering_rule = 'GWsTA';
tampering_type = 'erase';

silent = false; % If you don't want to see the progress output

% == Launching the runs
D = zeros(numel(M), numel(C));
E = zeros(numel(M), numel(C));
tperf = cputime(); % to show the total time elapsed later
for c=1:numel(C) % for each value of c
    cnetwork = logical(sparse([]));
    thriftymessages = logical(sparse([]));
    for m=1:numel(M) % and for each value of m, we will do a run
        % Launch the run
        if m == 1
            [cnetwork, thriftymessages, density] = gbnn_learn('m', round(M(1, m)*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', C(1, c), 'Chi', Chi, 'silent', silent);
        else % Optimization trick: instead of relearning the whole network, we will reuse the previous network and just add more messages, this allows to decrease the learning time exponentially, rendering it constant (at each learning, the network will learn the same amount of messages: eg: iteration 1 will learn 1E5 messages, iteration 2 will learn 1E5 messages and reuse 1E5, which will totalize as 2E5, etc...)
            [cnetwork, s2, density] = gbnn_learn('cnetwork', cnetwork, ...
                                                        'm', round((M(1, m)-M(1,m-1))*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', C(1, c), 'Chi', Chi, ...
                                                        'silent', silent);
            thriftymessages = [thriftymessages ; s2]; % append new messages
        end

        error_rate = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                  'erasures', erasures(1,c), 'iterations', iterations, 'tampered_messages_per_test', tampered_messages_per_test, 'tests', tests, ...
                                                                                  'enable_guiding', enable_guiding, 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', propagation_rule, 'filtering_rule', filtering_rule, 'tampering_type', tampering_type, ...
                                                                                  'silent', silent);
        
        % Store the results
        D(m,c) = density;
        E(m,c) = error_rate;
        if ~silent; fprintf('-----------------------------\n\n'); end;
    end
end
aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs: %G seconds.\n'); aux.flushout(); % print total time elapsed

% == Plotting

% 1- Plot density with respect to number of stored messages
figure; hold on;
xlabel(sprintf('Number of stored messages (M) x %.1E', Mcoeff));
ylabel('Network density (d)');
for c=1:numel(C) % for each value of c (order of the clique) we willl print a different curve, with an automatically selected color and shape
    lstyle = linestylevec(c, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
    cur_plot = plot(M, D(:,c), sprintf('%s%s%s', lstyle, markerstylevec(c), colorvec(c))); % plot one line
    set(cur_plot, 'DisplayName', sprintf('c = %i', C(1,c))); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
end
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName')); % IMPORTANT: force refreshing to show the legend, else it won't show!

% 2- Plot error rate with respect to the number of stored messages
figure; hold on;
xlabel(sprintf('Number of stored messages (M) x %.1E', Mcoeff));
ylabel('Error rate');
for c=1:numel(C) % for each value of c (order of the clique) we willl print a different curve, with an automatically selected color and shape
    lstyle = linestylevec(c, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
    cur_plot = plot(M, E(:,c), sprintf('%s%s%s', lstyle, markerstylevec(c), colorvec(c))); % plot one line
    set(cur_plot, 'DisplayName', sprintf('c = %i', C(1,c))); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
end
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName')); % IMPORTANT: force refreshing to show the legend, else it won't show! Thank's to tmpearce.

% Print densities values and error rates
fprintf('Densities:\n'); disp(D);
fprintf('Error rates:\n'); disp(E);

% The end!
