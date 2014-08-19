% This example main file shows how to reproduce the figure 3 of the 2014 article but with concurrent messages. This is mainly used to analyze and compare the performances of the different filtering_rules when concurrency is enabled.

% Clear things up
clear all;
close all;

% Importing auxiliary functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% Preparing stuff to automate the plots
% This will allow us to automatically select a different color and shape for each curve
colorvec = 'krbgmc';
markerstylevec = '+o*.xsd^v><ph';
linestylevec = {'-' ; '--' ; ':' ; '-.'};

% Vars config, tweak the stuff here
M = [1:2:15 16 18 20]; % this is a vector because we will try several values of m (number of messages, which influences the density)
Mcoeff = 1E2;
miterator = zeros(1,numel(M)); %M/2;
c = 8;
l = 16;
Chi = 32;
erasures = 2;
iterations = 4; % for convergence
tampered_messages_per_test = 30;
tests = 1;

enable_guiding = false;
gamma_memory = 0;
threshold = 0;
propagation_rule = 'sum';
filtering_rule = {'ML', 'GWsTA', 'GWsTA', 'GWsTA', 'GWsTA'}; % this is a cell array (vector of strings) because we will try several different values of c (order of cliques)
tampering_type = 'erase';

residual_memory = 0;
concurrent_cliques = 2;
no_concurrent_overlap = true;
concurrent_successive = false;
concurrent_disequilibrium = [false, true, 3, 2, false];
filtering_rule_first_iteration = false;
filtering_rule_last_iteration = false;

statstries = 5; % retry n times with different networks to smooth the results

silent = false; % If you don't want to see the progress output

% == Launching the runs
D = zeros(numel(M), numel(filtering_rule));
E = zeros(numel(M), numel(filtering_rule));
TE = zeros(numel(M), 1); % theoretical error rate depends on: Chi, l, c, erasures, enable_guiding and of course the density (theoretical or real) and thus on any parameter that changes the network (thus as the number of messages m to learn)
ED = zeros(numel(M), numel(filtering_rule));
SM = zeros(numel(M), numel(filtering_rule));
MM = zeros(numel(M), numel(filtering_rule));

for t=1:statstries
    tperf = cputime(); % to show the total time elapsed later
    cnetwork = logical(sparse([]));
    thriftymessages = logical(sparse([]));
    for m=1:numel(M) % and for each value of m, we will do a run
        % Launch the run
        if m == 1
            [cnetwork, thriftymessages, density] = gbnn_learn('m', round(M(1, 1)*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, 'silent', silent);
        else % Optimization trick: instead of relearning the whole network, we will reuse the previous network and just add more messages, this allows to decrease the learning time exponentially, rendering it constant (at each learning, the network will learn the same amount of messages: eg: iteration 1 will learn 1E5 messages, iteration 2 will learn 1E5 messages and reuse 1E5, which will totalize as 2E5, etc...)
            [cnetwork, s2, density] = gbnn_learn('cnetwork', cnetwork, ...
                                                        'm', round((M(1, m)-M(1,m-1))*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                        'silent', silent);
            thriftymessages = [thriftymessages ; s2]; % append new messages
        end

        counter = 1;
        for f=1:numel(filtering_rule)
            fr = filtering_rule{f}; % need to prepare beforehand because of MatLab, can't do it in one command...
            if strcmpi(fr, 'ML')
                iterations_bak = iterations;
                iterations = 1;
            end
            [error_rate, theoretical_error_rate, test_stats] = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                  'erasures', erasures, 'iterations', iterations, 'tampered_messages_per_test', tampered_messages_per_test, 'tests', tests, ...
                                                                                  'enable_guiding', enable_guiding, 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', propagation_rule, 'filtering_rule', fr, 'tampering_type', tampering_type, ...
                                                                                  'residual_memory', residual_memory, 'concurrent_cliques', concurrent_cliques, 'no_concurrent_overlap', no_concurrent_overlap, 'concurrent_successive', concurrent_successive, 'filtering_rule_first_iteration', filtering_rule_first_iteration, 'filtering_rule_last_iteration', filtering_rule_last_iteration, ...
                                                                                  'concurrent_disequilibrium', concurrent_disequilibrium(f), ...
                                                                                  'silent', silent);
            if strcmpi(fr, 'ML')
                iterations = iterations_bak;
            end

            % Store the results
            D(m,counter) = D(m,counter) + density;
            E(m,counter) = E(m,counter) + error_rate;
            TE(m) = theoretical_error_rate;
            ED(m, counter) = ED(m, counter) + test_stats.error_distance;
            SM(m, counter) = SM(m, counter) + test_stats.similarity_measure;
            MM(m, counter) = MM(m, counter) + test_stats.matching_measure;
            if ~silent; fprintf('-----------------------------\n\n'); end;
            
            counter = counter + 1;
        end
    end
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs: %G seconds.\n'); aux.flushout(); % print total time elapsed
end
% Normalizing errors rates by calculating the mean error for all tries
D = D ./ statstries;
E = E ./ statstries;
ED = ED ./ statstries;
SM = SM ./ statstries;
MM = MM ./ statstries;
printf('END of all tests!\n'); aux.flushout();


% == Plotting

% -- Plot error rate with respect to the density (or number of messages stored) and a few other parameters
figure; hold on;
xlabel(sprintf('Number of stored messages (M) x %.1E', Mcoeff));
ylabel('Retrieval Error Rate');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for f=1:numel(filtering_rule) % for each different filtering rule and whether there is guiding or not, we willl print a different curve, with an automatically selected color and shape
    coloridx = mod(f-1, numel(colorvec))+1; % change color per filtering rule
    lstyleidx = mod(counter-1, numel(linestylevec))+1; % change line style ...
    mstyleidx = mod(counter-1, numel(markerstylevec))+1; % and change marker style per plot

    lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
    cur_plot = plot(M, E(:,counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line

    fr = filtering_rule(1,f); fr = fr{1};
    plot_title = sprintf('%s', fr);
    if enable_guiding
        plot_title = strcat(plot_title, sprintf(' - Guided'));
    else
        plot_title = strcat(plot_title, sprintf(' - Blind'));
    end
    if concurrent_disequilibrium(f)
        plot_title = strcat(plot_title, sprintf(' - Diseq'));
        if concurrent_disequilibrium(f) > 1
            plot_title = strcat(plot_title, sprintf(' type %i', concurrent_disequilibrium(f)));
        end
    end
    set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

    counter = counter + 1;
end

% Plot theoretical error rates
counter = counter + 1;
coloridx = mod(counter, numel(colorvec))+1;
for g=1:numel(enable_guiding)
    lstyleidx = mod(counter+g-1, numel(linestylevec))+1;
    mstyleidx = mod(counter+g-1, numel(markerstylevec))+1;

    lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
    cur_plot = plot(M, TE(:,g), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line

    plot_title = '';
    if enable_guiding
        plot_title = strcat(plot_title, sprintf('Guided'));
    else
        plot_title = strcat(plot_title, sprintf('Blind'));
    end
    plot_title = strcat(plot_title, ' (Theo.)');
    set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
end

% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName')); % IMPORTANT: force refreshing to show the legend, else it won't show!


% -- Plot matching_measure and other stats
figure; hold on;
xlabel(sprintf('Number of stored messages (M) x %.1E', Mcoeff));
ylim([0 1]);
f = 2;
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)

fr = filtering_rule(1,f); fr = fr{1};
if concurrent_disequilibrium(f); fr = strcat(fr, ' diseq'); end;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(M, 1-MM(:,f), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, 'DisplayName', strcat(fr, ' - mismatching measure')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(M, 1-SM(:,f), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, 'DisplayName', strcat(fr, ' - dissimilarity measure')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(M, E(:,f), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, 'DisplayName', strcat(fr, ' - error rate')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(M, ED(:,f), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, 'DisplayName', strcat(fr, ' - error distance')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName')); % IMPORTANT: force refreshing to show the legend, else it won't show!



% Print densities values and error rates
fprintf('Densities:\n'); disp(D);
fprintf('Error rates:\n'); disp(E);
fprintf('Theoretical error rates:\n'); disp(TE);

% The end!
