% This example main file draws useful plots to analyze the evolution of the network and its error rates with different number of concurrent messages

% Clear things up
clear all;
close all;

% Importing auxiliary functions
% source('gbnn_aux.m'); % does not work with MatLab, only Octave...
aux = gbnn_aux; % works with both MatLab and Octave

% Preparing stuff to automate the plots
% This will allow us to automatically select a different color and shape for each curve
colorvec = 'rkgbmc';
markerstylevec = '+o*.xsd^v><ph';
linestylevec = {'-' ; '--' ; ':' ; '-.'};

% Vars config, tweak the stuff here
M = [0.1 1:1:11 15:5:45]; % this is a vector because we will try several values of m (number of messages, which influences the density)
Mcoeff = 1E3;
miterator = zeros(1,numel(M)); %M/2;
c = 12;
l = 32;
Chi = 64;
erasures = 3;
iterations = 4; % for convergence
tampered_messages_per_test = 100;
tests = 1;

enable_guiding = [false, true]; % here too, we will try with and without the guiding mask
gamma_memory = 1;
threshold = 0;
propagation_rule = 'sum'; % TODO: not implemented yet, please always set 0 here
filtering_rule = {'GWsTA'}; % this is a cell array (vector of strings) because we can try several different filtering rules
tampering_type = 'erase';

residual_memory = 0;
concurrent_cliques = 1:3;
concurrent_disequilibrium = 1; % 1 for superscore mode, 2 for one fanal erasure, 3 for nothing at all just trying to decode one clique at a time without any trick, 0 to disable
no_concurrent_overlap = false;
concurrent_successive = false;
filtering_rule_first_iteration = false;
filtering_rule_last_iteration = false;

% Training with auxiliary support network parameters (optional)
training = false; % switch to true to do the training step (disambiguation of conflicting memories)
c2 = 2; % should be << c
l2 = 1; % can be set to 1
Chi2 = Chi;
trainingbatchs = 2;
no_auxiliary_propagation = false;

% Plot tweaking
statstries = 3; % retry n times with different networks to average (and thus smooth) the results
smooth_factor = 2; % interpolate more points to get smoother curves. Set to 1 to avoid smoothing (and thus plot only the point of the real samples).
smooth_method = 'cubic'; % use PCHIP or cubic to avoid interpolating into negative values as spline does
plot_curves_params = { 'markersize', 10, ...
                                            'linewidth', 1 ...
                                            };
plot_axis_params = { 'linewidth', 1, ...
                                      'tickdir', 'out', ...
                                      'ticklength', [0.01, 0.01] ...
                                      };
plot_text_params = { 'FontSize', 12, ... % in points
                                       'FontName', 'Helvetica' ...
                                       };

plot_theo = true; % plot theoretical error rates?
silent = false; % If you don't want to see the progress output

% == Launching the runs
D = zeros(numel(M), numel(filtering_rule)*numel(enable_guiding)*numel(concurrent_cliques));
E = zeros(numel(M), numel(filtering_rule)*numel(enable_guiding)*numel(concurrent_cliques));
ED = zeros(numel(M), numel(filtering_rule)*numel(enable_guiding)*numel(concurrent_cliques));
TE = zeros(numel(M), numel(enable_guiding)*numel(concurrent_cliques)); % theoretical error rate depends on: Chi, l, c, erasures, enable_guiding and of course the density (theoretical or real) and thus on any parameter that changes the network (thus as the number of messages m to learn)
EC = zeros(numel(M), numel(filtering_rule)*numel(enable_guiding)*numel(concurrent_cliques));

for t=1:statstries
    tperf = cputime(); % to show the total time elapsed later
    cnetwork = logical(sparse([]));
    thriftymessages = logical(sparse([]));
    for m=1:numel(M) % and for each value of m, we will do a run

        % Learning phase
        if m == 1
            [cnetwork, s2, density] = gbnn_learn('m', round(M(1, 1)*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, 'silent', silent);
        else % Optimization trick: instead of relearning the whole network, we will reuse the previous network and just add more messages, this allows to decrease the learning time exponentially, rendering it constant (at each learning, the network will learn the same amount of messages: eg: iteration 1 will learn 1E5 messages, iteration 2 will learn 1E5 messages and reuse 1E5, which will totalize as 2E5, etc...)
            [cnetwork, s2, density] = gbnn_learn('cnetwork', cnetwork, ...
                                                        'm', round((M(1, m)-M(1,m-1))*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                        'silent', silent);
            
        end
        thriftymessages = [thriftymessages ; s2]; % append new messages

        % Training phase (optional)
        if training
            cnetwork = gbnn_train('cnetwork', cnetwork, 'thriftymessagestest', s2, 'l', l2, 'c', c2, 'Chi', Chi2, 'tampered_messages_per_test', tampered_messages_per_test, 'tests', trainingbatchs, 'no_auxiliary_propagation', no_auxiliary_propagation);
        end

        % Testing phase
        counter = 1;
        for f=1:numel(filtering_rule)
            tecounter = 1;
            for cc=1:numel(concurrent_cliques)
                for g=1:numel(enable_guiding)
                    fr = filtering_rule(1,f); fr = fr{1}; % need to prepare beforehand because of MatLab, can't do it in one command...
                    [error_rate, theoretical_error_rate, test_stats] = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                          'erasures', erasures, 'iterations', iterations, 'tampered_messages_per_test', tampered_messages_per_test, 'tests', tests, ...
                                                                                          'enable_guiding', enable_guiding(1,g), 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', propagation_rule, 'filtering_rule', fr, 'tampering_type', tampering_type, 'filtering_rule_first_iteration', filtering_rule_first_iteration, 'filtering_rule_last_iteration', filtering_rule_last_iteration, ...
                                                                                          'residual_memory', residual_memory, 'concurrent_cliques', concurrent_cliques(1,cc), 'concurrent_disequilibrium', concurrent_disequilibrium, 'no_concurrent_overlap', no_concurrent_overlap, 'concurrent_successive', concurrent_successive, ...
                                                                                          'silent', silent);

                    % Store the results
                    D(m,counter) = D(m,counter) + density;
                    E(m,counter) = E(m,counter) + error_rate;
                    ED(m, counter) = ED(m, counter) + test_stats.error_distance;
                    TE(m, tecounter) = theoretical_error_rate;
                    EC(m, counter) = EC(m, counter) + test_stats.concurrent_unbiased_error_rate;
                    if ~silent; fprintf('-----------------------------\n\n'); end;
                    
                    counter = counter + 1;
                    tecounter = tecounter + 1;
                end
            end
        end
    end
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs: %G seconds.\n'); aux.flushout(); % print total time elapsed
end
% Normalizing errors rates by calculating the mean error for all tries
D = D ./ statstries;
E = E ./ statstries;
ED = ED ./ statstries;
EC = EC ./ statstries;
printf('END of all tests!\n'); aux.flushout();


% == Plotting

% -- First interpolate data points to get smoother curves
% Note: if smooth_factor == 1 then these commands won't change the data points nor add more.
nsamples = numel(M);
M_interp = interp1(1:nsamples, M, linspace(1, nsamples, nsamples*smooth_factor), smooth_method);
D_interp = interp1(1:nsamples, D(:,1), linspace(1, nsamples, nsamples*smooth_factor), smooth_method);
E_interp = interp1(D(:,1), E, D_interp, smooth_method);
TE_interp = interp1(D(:,1), TE, D_interp, smooth_method);
ED_interp = interp1(D(:,1), ED, D_interp, smooth_method);;
EC_interp = interp1(D(:,1), EC, D_interp, smooth_method);

% -- Plot error rate with respect to the density (or number of messages stored) and a few other parameters
figure; hold on;
xlabel(sprintf('(Bottom) Density  -- (Top) Number of stored messages (M) x%.1E', Mcoeff));
ylabel('Retrieval Error Rate');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for f=1:numel(filtering_rule) % for each different filtering rule and whether there is guiding or not, we willl print a different curve, with an automatically selected color and shape
    coloridx = mod(f-1, numel(colorvec))+1; % change color per filtering rule
    counterstyle = 1; % use another counter for styles, so that each curve will get the exact same style for each set of parameters
    for cc=1:numel(concurrent_cliques)
        for g=1:numel(enable_guiding)
            lstyleidx = mod(counterstyle-1, numel(linestylevec))+1; % change line style ...
            mstyleidx = mod(counterstyle-1, numel(markerstylevec))+1; % and change marker style per plot

            lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...

            % Prepare the legend
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
            plot_title = strcat(plot_title, sprintf(' - %i it', iterations(f)));

            % Draw the curves
            % => Error rate
            cur_plot = plot(D_interp, E_interp(:,counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
            set(cur_plot, plot_curves_params{:}); % additional plot style
            set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

            counter = counter + 1;
            counterstyle = counterstyle + 1;
        end
    end
end
% Plot theoretical error rates
if plot_theo
    coloridx = mod(counter, numel(colorvec))+1; % change color for theoretical errors
    tecounter = 1;
    for cc=1:numel(concurrent_cliques)
        for g=1:numel(enable_guiding)
            lstyleidx = mod(tecounter-1, numel(linestylevec))+1;
            mstyleidx = mod(tecounter-1, numel(markerstylevec))+1;

            lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
            cur_plot = plot(D_interp, TE_interp(:,tecounter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
            set(cur_plot, plot_curves_params{:}); % additional plot style

            plot_title = '';
            if concurrent_cliques(1,cc) == 1
                plot_title = strcat(plot_title, sprintf('no cc'));
            else
                plot_title = strcat(plot_title, sprintf('cc = %i', concurrent_cliques(1, cc)));
            end
            if enable_guiding(1,g)
                plot_title = strcat(plot_title, sprintf(' - Guided'));
            else
                plot_title = strcat(plot_title, sprintf(' - Blind'));
            end
            plot_title = strcat(plot_title, ' (Theo.)');
            set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

            tecounter = tecounter + 1;
        end
    end
end

% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName'), 'location', 'southeast'); % IMPORTANT: force refreshing to show the legend, else it won't show!
legend('boxoff');
% Add secondary axis on the top of the figure to show the number of messages
aux.add_2nd_xaxis(D(:,1), M, sprintf('x%.1E', Mcoeff), '%g', 0);
xlim([0 max(D(:,1))]); % adjust x axis zoom
% Adjust axis drawing style
set( gca(), plot_axis_params{:} );
% Adjust text style
set([gca; findall(gca, 'Type','text')], plot_text_params{:});


% -- Plot concurrent unbiased error rate with respect to the density
figure; hold on;
xlabel(sprintf('(Bottom) Density  -- (Top) Number of stored messages (M) x%.1E', Mcoeff));
ylabel('Unbiased Retrieval Error Rate per clique');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for f=1:numel(filtering_rule) % for each different filtering rule and whether there is guiding or not, we willl print a different curve, with an automatically selected color and shape
    coloridx = mod(f-1, numel(colorvec))+1; % change color per filtering rule
    counterstyle = 1; % use another counter for styles, so that each curve will get the exact same style for each set of parameters
    for cc=1:numel(concurrent_cliques)
        for g=1:numel(enable_guiding)
            lstyleidx = mod(counterstyle-1, numel(linestylevec))+1; % change line style ...
            mstyleidx = mod(counterstyle-1, numel(markerstylevec))+1; % and change marker style per plot

            lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...

            % Prepare the legend
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
            plot_title = strcat(plot_title, sprintf(' - %i it', iterations(f)));

            % Draw the curves
            % => Error rate
            cur_plot = plot(D_interp, EC_interp(:,counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
            set(cur_plot, plot_curves_params{:}); % additional plot style
            set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

            counter = counter + 1;
            counterstyle = counterstyle + 1;
        end
    end
end
% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName'), 'location', 'southeast'); % IMPORTANT: force refreshing to show the legend, else it won't show!
legend('boxoff');
% Add secondary axis on the top of the figure to show the number of messages
aux.add_2nd_xaxis(D(:,1), M, sprintf('x%.1E', Mcoeff), '%g', 0);
xlim([0 max(D(:,1))]); % adjust x axis zoom
% Adjust axis drawing style
set( gca(), plot_axis_params{:} );
% Adjust text style
set([gca; findall(gca, 'Type','text')], plot_text_params{:});


% -- Plot error distance
figure; hold on;
xlabel(sprintf('(Bottom) Density  -- (Top) Number of stored messages (M) x%.1E', Mcoeff));
ylabel('Mean Error Distance per message (above y=1 is random)');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for f=1:numel(filtering_rule) % for each different filtering rule and whether there is guiding or not, we willl print a different curve, with an automatically selected color and shape
    coloridx = mod(f-1, numel(colorvec))+1; % change color per filtering rule
    counterstyle = 1; % use another counter for styles, so that each curve will get the exact same style for each set of parameters
    for cc=1:numel(concurrent_cliques)
        for g=1:numel(enable_guiding)
            lstyleidx = mod(counterstyle-1, numel(linestylevec))+1; % change line style ...
            mstyleidx = mod(counterstyle-1, numel(markerstylevec))+1; % and change marker style per plot

            lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...

            % Prepare the legend
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
            if concurrent_disequilibrium(f)
                plot_title = strcat(plot_title, sprintf(' - Diseq type %i', concurrent_disequilibrium(f)));
            else
                plot_title = strcat(plot_title, sprintf(' - No diseq'));
            end
            plot_title = strcat(plot_title, sprintf(' - %i it', iterations(f)));

            % Draw the curves
            % => Error distance
            cur_plot = plot(D_interp, ED_interp(:,counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
            set(cur_plot, plot_curves_params{:}); % additional plot style
            set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

            counter = counter + 1;
            counterstyle = counterstyle + 1;
        end
    end
end
% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName'), 'location', 'northwest'); % IMPORTANT: force refreshing to show the legend, else it won't show!
legend('boxoff');
% Add secondary axis on the top of the figure to show the number of messages
aux.add_2nd_xaxis(D(:,1), M, sprintf('x%.1E', Mcoeff), '%g', 0);
xlim([0 max(D(:,1))]); % adjust x axis zoom
% Adjust axis drawing style
set( gca(), plot_axis_params{:} );
% Adjust text style
set([gca; findall(gca, 'Type','text')], plot_text_params{:});


% Print densities values and error rates
fprintf('Densities:\n'); disp(D);
fprintf('Error rates:\n'); disp(E);
fprintf('Theoretical error rates:\n'); disp(TE);
fprintf('Euclidian error distances:\n'); disp(ED);

% The end!
