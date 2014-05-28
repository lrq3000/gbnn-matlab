% This example main file shows how to do a batch of runs to produce figures like those in the 2014 article.

% Clear things up
clear all;
close all;

% Importing auxiliary functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% Preparing stuff to automate the plots
% This will allow us to automatically select a different color and shape for each curve
colorvec = 'krgbmcw';
markerstylevec = '+o*.xsd^v><ph';
linestylevec = {'-' ; '--' ; ':' ; '-.'};

% Vars config, tweak the stuff here
C = [8 12 16 20]; % this is a vector because we will try several different values of c (order of cliques)
M = 1:1:10; % and different values of m (number of messages, which influences the density)
Mcoeff = 1E5;
miterator = zeros(1,numel(M)); %M/2;
l = 64;
Chi = 100;
erasures = C/2;
iterations = 4;
tampered_messages_per_test = 200;
tests = 1;

guiding_mask = true;
gamma_memory = 0;
threshold = 0;
propagation_rule = 'sum'; % TODO: not implemented yet, please always set 0 here
filtering_rule = 'GkWTA';
tampering_type = 'erase';

residual_memory = 0;
variable_length = false;
concurrent_cliques = 2;
GWTA_first_iteration = false;
GWTA_last_iteration = false;

silent = false; % If you don't want to see the progress output

% == Launching the runs
D = zeros(numel(M), numel(C));
E = zeros(numel(M), numel(C));
tperf = cputime(); % to show the total time elapsed later
for c=1:numel(C) % for each value of c
    for m=1:numel(M) % and for each value of m, we will do a run
        % Launch the run
        [~, density, error_rate] = gbnn([], [], ...
                                                    M(1, m)*Mcoeff, miterator(1,m), l, C(1, c), erasures(1,c), iterations, tampered_messages_per_test, tests, ...
                                                    Chi, guiding_mask, gamma_memory, threshold, propagation_rule, filtering_rule, tampering_type, ...
                                                    residual_memory, variable_length, concurrent_cliques, GWTA_first_iteration, GWTA_last_iteration, ...
                                                    silent);
        %clear all; gbnn([], [], 6, 0, 4, 3, 1, 4, 3, 1, 5, true, 3, 2, 0, 0, 0, false, 2); % to call directly one run with an example small random test set, useful for debugging, and don't forget to clear all; before!

        % Store the results
        D(m,c) = density;
        E(m,c) = error_rate;
        fprintf('-----------------------------\n\n');
    end
end
aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs: %g seconds.\n'); aux.flushout(); % print total time elapsed

% == Plotting

% 1- Plot density with respect to number of stored messages
figure; hold on;
xlabel(sprintf('Number of stored messages (M) x %.1e', Mcoeff));
ylabel('Network density (d)');
for c=1:numel(C) % for each value of c (order of the clique) we willl print a different curve, with an automatically selected color and shape
    lstyle = linestylevec(c, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
    cur_plot = plot(M, D(:,c), sprintf('%s%s%s', lstyle, markerstylevec(c), colorvec(c))); % plot one line
    set(cur_plot, 'DisplayName', sprintf('c = %i', C(1,c))); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
end
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName')); % IMPORTANT: force refreshing to show the legend, else it won't show!

% 2- Plot error rate with respect to the number of stored messages
figure; hold on;
xlabel(sprintf('Number of stored messages (M) x %.1e', Mcoeff));
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
