% This example main file shows how to do a batch of runs to produce figures like those in the 2014 article.

% Clear things up
clear all;
close all;

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
propagation_rule = 'sum'; % TODO: not implemented yet, please always set 0 here
filtering_rule = 'GWsTA';
tampering_type = 'erase';

residual_memory = 0;
variable_length = false;
concurrent_cliques = 1;
no_concurrent_overlap = false;
GWTA_first_iteration = false;
GWTA_last_iteration = false;

silent = false; % If you don't want to see the progress output

% == Launching the runs
D = zeros(numel(M), numel(C));
E = zeros(numel(M), numel(C));
tperf = cputime(); % to show the total time elapsed later
for c=1:numel(C) % for each value of c
    network = logical(sparse([]));
    sparsemessages = logical(sparse([]));
    for m=1:numel(M) % and for each value of m, we will do a run
        % Launch the run
        if m == 1
            [network, sparsemessages, density] = gbnn_learn([], M(1, m)*Mcoeff, miterator(1,m), l, C(1, c), Chi, variable_length, silent);
        else % Optimization trick: instead of relearning the whole network, we will reuse the previous network and just add more messages, this allows to decrease the learning time exponentially, rendering it constant (at each learning, the network will learn the same amount of messages: eg: iteration 1 will learn 1E5 messages, iteration 2 will learn 1E5 messages and reuse 1E5, which will totalize as 2E5, etc...)
            [network, s2, density] = gbnn_learn(network, ...
                                                        (M(1, m)-M(1,m-1))*Mcoeff, miterator(1,m), l, C(1, c), Chi, ...
                                                        variable_length, ...
                                                        silent);
            sparsemessages = [sparsemessages ; s2]; % append new messages
        end

        error_rate = gbnn_test(network, sparsemessages, ...
                                                                                  l, C(1,c), Chi, ...
                                                                                  erasures(1,c), iterations, tampered_messages_per_test, tests, ...
                                                                                  enable_guiding, gamma_memory, threshold, propagation_rule, filtering_rule, tampering_type, ...
                                                                                  residual_memory, variable_length, concurrent_cliques, no_concurrent_overlap, GWTA_first_iteration, GWTA_last_iteration, ...
                                                                                  silent);
        
        % Store the results
        D(m,c) = density;
        E(m,c) = error_rate;
        fprintf('-----------------------------\n\n');
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
