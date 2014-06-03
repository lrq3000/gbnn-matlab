% This example main file shows how to reproduce the figure 3 of the 2014 article.

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
M = 0.5:0.5:4.5; % this is a vector because we will try several values of m (number of messages, which influences the density)
Mcoeff = 1E5;
miterator = zeros(1,numel(M)); %M/2;
c = 12;
l = 64;
Chi = 100;
erasures = 3;
iterations = 4;
tampered_messages_per_test = 100;
tests = 1;

enable_guiding = [false, true]; % here too, we will try with and without the guiding mask
gamma_memory = 1;
threshold = 0;
propagation_rule = 'sum'; % TODO: not implemented yet, please always set 0 here
filtering_rule = {'GWsTA'}; % this is a cell array (vector of strings) because we will try several different values of c (order of cliques)
tampering_type = 'erase';

residual_memory = 0;
variable_length = false;
concurrent_cliques = 1:4;
GWTA_first_iteration = false;
GWTA_last_iteration = false;

silent = false; % If you don't want to see the progress output

% == Launching the runs
D = zeros(numel(M), numel(filtering_rule)*numel(enable_guiding)*numel(concurrent_cliques));
E = zeros(numel(M), numel(filtering_rule)*numel(enable_guiding)*numel(concurrent_cliques));
TE = zeros(numel(M), numel(enable_guiding)); % theoretical error rate depends on: Chi, l, c, erasures, enable_guiding and of course the density (theoretical or real) and thus on any parameter that changes the network (thus as the number of messages m to learn)
tperf = cputime(); % to show the total time elapsed later
network = logical(sparse([]));
sparsemessages = logical(sparse([]));
for m=1:numel(M) % and for each value of m, we will do a run
    % Launch the run
    if m == 1
        [network, sparsemessages, density] = gbnn_learn([], M(1, 1)*Mcoeff, miterator(1,m), l, c, Chi, variable_length, silent);
    else % Optimization trick: instead of relearning the whole network, we will reuse the previous network and just add more messages, this allows to decrease the learning time exponentially, rendering it constant (at each learning, the network will learn the same amount of messages: eg: iteration 1 will learn 1E5 messages, iteration 2 will learn 1E5 messages and reuse 1E5, which will totalize as 2E5, etc...)
        [network, s2, density] = gbnn_learn(network, ...
                                                    (M(1, m)-M(1,m-1))*Mcoeff, miterator(1,m), l, c, Chi, ...
                                                    variable_length, ...
                                                    silent);
        sparsemessages = [sparsemessages ; s2]; % append new messages
    end

    counter = 1;
    for f=1:numel(filtering_rule)
        for cc=1:numel(concurrent_cliques)
            for g=1:numel(enable_guiding)
                fr = filtering_rule(1,f); fr = fr{1}; % need to prepare beforehand because of MatLab, can't do it in one command...
                [error_rate, theoretical_error_rate] = gbnn_test(network, sparsemessages, ...
                                                                                      l, c, Chi, ...
                                                                                      erasures, iterations, tampered_messages_per_test, tests, ...
                                                                                      enable_guiding(1,g), gamma_memory, threshold, propagation_rule, fr, tampering_type, ...
                                                                                      residual_memory, variable_length, concurrent_cliques(1,cc), GWTA_first_iteration, GWTA_last_iteration, ...
                                                                                      silent);

                % Store the results
                D(m,counter) = density;
                E(m,counter) = error_rate;
                TE(m, g) = theoretical_error_rate;
                fprintf('-----------------------------\n\n');
                
                counter = counter + 1;
            end
        end
    end
end
aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs: %G seconds.\n'); aux.flushout(); % print total time elapsed

% == Plotting

% Plot density with respect to number of stored messages
figure; hold on;
xlabel(sprintf('Number of stored messages (M) x %.1E', Mcoeff));
ylabel('Retrieval Error Rate');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for f=1:numel(filtering_rule) % for each different filtering rule and whether there is guiding or not, we willl print a different curve, with an automatically selected color and shape
    coloridx = mod(f-1, numel(colorvec))+1; % change color per filtering rule
    for cc=1:numel(concurrent_cliques)
        for g=1:numel(enable_guiding)
            lstyleidx = mod(counter-1, numel(linestylevec))+1; % change line style ...
            mstyleidx = mod(counter-1, numel(markerstylevec))+1; % and change marker style per plot

            lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
            cur_plot = plot(M, E(:,counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line

            fr = filtering_rule(1,f); fr = fr{1};
            plot_title = sprintf('%s', fr);
            if concurrent_cliques(1,cc) == 1
                plot_title = strcat(plot_title, sprintf(' - no cc'));
            else
                plot_title = strcat(plot_title, sprintf(' - cc = %i', concurrent_cliques(1, cc)));
            end
            if enable_guiding(1,g)
                plot_title = strcat(plot_title, sprintf(' - Guided'));
            else
                plot_title = strcat(plot_title, sprintf(' - Blind'));
            end
            set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

            counter = counter + 1;
        end
    end
end

% Plot theoretical error rates
counter2 = 1;
counter = counter + 1;
for cc=1:numel(concurrent_cliques)
    coloridx = mod(counter+cc-1, numel(colorvec))+1;
    for g=1:numel(enable_guiding)
        lstyleidx = mod(counter+counter2-1, numel(linestylevec))+1;
        mstyleidx = mod(counter+counter2-1, numel(markerstylevec))+1;

        lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        cur_plot = plot(M, TE(:,g), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line

        plot_title = '';
        if concurrent_cliques(1,cc) == 1
            plot_title = strcat(plot_title, sprintf(' - no cc'));
        else
            plot_title = strcat(plot_title, sprintf(' - cc = %i', concurrent_cliques(1, cc)));
        end
        if enable_guiding(1,g)
            plot_title = strcat(plot_title, sprintf('Guided'));
        else
            plot_title = strcat(plot_title, sprintf('Blind'));
        end
        plot_title = strcat(plot_title, ' (Theo.)');
        set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

        counter2 = counter2 + 1;
    end
end

% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName')); % IMPORTANT: force refreshing to show the legend, else it won't show!

% Print densities values and error rates
fprintf('Densities:\n'); disp(D);
fprintf('Error rates:\n'); disp(E);
fprintf('Theoretical error rates:\n'); disp(TE);

% The end!
