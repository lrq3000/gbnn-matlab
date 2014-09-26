% Try to reproduce the figure obtained by Ehsan on taggued networks

% Clear things up
clear all;
close all;

% Importing auxiliary functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% Preparing stuff to automate the plots
% This will allow us to automatically select a different color and shape for each curve
colorvec = 'rgbmc';
markerstylevec = '+o*.xsd^v><ph';
linestylevec = {'-' ; '--' ; ':' ; '-.'};

% Vars config, tweak the stuff here
M = 0.5:1:4.5; % this is a vector because we will try several values of m (number of messages, which influences the density)
Mcoeff = 1E3;
miterator = zeros(1,numel(M)); %M/2;
c = 8;
l = 64;
Chi = c;
erasures = floor(c/2);
iterations = 4; % for convergence
iterations_taggued = 4;
tampered_messages_per_test = 100;
tests = 1;

enable_guiding = false; % here too, we will try with and without the guiding mask
gamma_memory = 0;
threshold = 0;
propagation_rule = 'sum'; % TODO: not implemented yet, please always set 0 here
filtering_rule = 'GWTA'; % this is a cell array (vector of strings) because we will try several different values of c (order of cliques)
tampering_type = 'erase';

residual_memory = 0;
filtering_rule_first_iteration = false;
filtering_rule_last_iteration = false;

silent = false; % If you don't want to see the progress output
thdraw = false;

% == Launching the runs
D = zeros(numel(M), 1);
E = zeros(numel(M), 1);
EHERR1 = zeros(numel(M), 1);
EHERR2 = zeros(numel(M), 1);
EHERR3 = zeros(numel(M), 1);
TE = zeros(numel(M), 1); % theoretical error rate depends on: Chi, l, c, erasures, enable_guiding and of course the density (theoretical or real) and thus on any parameter that changes the network (thus as the number of messages m to learn)
tperf = cputime(); % to show the total time elapsed later
cnetwork = logical(sparse([]));
thriftymessages = logical(sparse([]));
for m=1:numel(M) % and for each value of m, we will do a run
    % Launch no tag network
    if m == 1
        [cnetwork, thriftymessages, density] = gbnn_learn('m', round(M(1, 1)*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, 'silent', silent);
    else % Optimization trick: instead of relearning the whole network, we will reuse the previous network and just add more messages, this allows to decrease the learning time exponentially, rendering it constant (at each learning, the network will learn the same amount of messages: eg: iteration 1 will learn 1E5 messages, iteration 2 will learn 1E5 messages and reuse 1E5, which will totalize as 2E5, etc...)
        [cnetwork, s2, density] = gbnn_learn('cnetwork', cnetwork, ...
                                                    'm', round((M(1, m)-M(1,m-1))*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                    'silent', silent);
        thriftymessages = [thriftymessages ; s2]; % append new messages
    end

    [error_rate, theoretical_error_rate] = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                          'erasures', erasures, 'iterations', iterations, 'tampered_messages_per_test', tampered_messages_per_test, 'tests', tests, ...
                                                                          'enable_guiding', enable_guiding, 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', propagation_rule, 'filtering_rule', filtering_rule, 'tampering_type', tampering_type, ...
                                                                          'residual_memory', residual_memory, 'filtering_rule_first_iteration', filtering_rule_first_iteration, 'filtering_rule_last_iteration', filtering_rule_last_iteration, ...
                                                                          'silent', silent);

    % Store the results
    D(m) = density;
    E(m) = error_rate;
    TE(m) = theoretical_error_rate;
    if ~silent; fprintf('-----------------------------\n\n'); end;

    % Launch Ehsan network
    [err1, err2, err3] = tagged_clique(c, l, round(M(m)*Mcoeff), iterations_taggued);
    EHERR1(m) = err1;
    EHERR2(m) = err2;
    EHERR3(m) = err3;
end
aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs: %G seconds.\n'); aux.flushout(); % print total time elapsed

% == Plotting

% Plot error rate with respect to the density (or number of messages stored) and a few other parameters
figure; hold on;
xlabel(sprintf('Number of stored messages (M) x %.1E', Mcoeff));
ylabel('Retrieval Error Rate');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for f=4:-1:1 % for each different filtering rule and whether there is guiding or not, we willl print a different curve, with an automatically selected color and shape
    coloridx = mod(f-1, numel(colorvec))+1; % change color per filtering rule
    lstyleidx = mod(counter-1, numel(linestylevec))+1; % change line style ...
    mstyleidx = mod(counter-1, numel(markerstylevec))+1; % and change marker style per plot

    lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...

    datatodraw = [];
    plot_title = '';
    if f == 1
        plot_title = 'No tag - Err:real-error';
        datatodraw = E;
    elseif f == 2
        plot_title = 'M tags - Err:tag-recovered';
        datatodraw = EHERR1;
    elseif f == 3
        plot_title = 'M tags - Err:message-length';
        datatodraw = EHERR2;
    elseif f == 4
        plot_title = 'M tags - Err:real-error';
        datatodraw = EHERR3;
    else
        error('f outside score, dont know what error to point to!\n');
    end
    cur_plot = plot(M, datatodraw, sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line

    plot_title = strcat(plot_title, sprintf(' - %s', filtering_rule));
    if enable_guiding
        plot_title = strcat(plot_title, sprintf(' - Guided'));
    else
        plot_title = strcat(plot_title, sprintf(' - Blind'));
    end
    set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

    counter = counter + 1;
end

% Plot theoretical error rates
if thdraw
    coloridx = mod(counter, numel(colorvec))+1;
    lstyleidx = mod(counter, numel(linestylevec))+1;
    mstyleidx = mod(counter, numel(markerstylevec))+1;

    lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
    cur_plot = plot(M, TE, sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line

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

% Print densities values and error rates
fprintf('Non-taggued network:\n');
fprintf('Densities:\n'); disp(D);
fprintf('Error rates:\n'); disp(E);
fprintf('Theoretical error rates:\n'); disp(TE);
fprintf('Taggued network with Ehsan error metrics:\n'); disp(EHERR1); disp(EHERR2); disp(EHERR3);

% The end!
