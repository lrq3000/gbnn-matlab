% Overlays network: Behrooz vs overlays theoretical benchmark to compare the theoretical error rate with the lost rate. Please use Octave >= 3.8.1 for reasonable performances!
% DEBUG MODE: this script (a variation of fig3tags1ehsan_th_evol.m) also outputs a second plot to show the various error measures, to better define what types of errors contribute the most to the tagged network error rate.

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
colorvec = 'rgbmc';
markerstylevec = '+o*.xsd^v><ph';
linestylevec = {'-' ; '--' ; ':' ; '-.'};

% Vars config, tweak the stuff here
M = [0.5:0.25:4 5:1:6]; % this is a vector because we will try several values of m (number of messages, which influences the density)
%M = [0.005 5.1]; % to test both limits to check that the range is OK, the first point must be near 0 and the second point must be near 1, at least for one of the curves
Mcoeff = 1E3;
miterator = zeros(1,numel(M)); %M/2;
c = 8;
l = 16;
Chi = 16;
erasures = floor(c/2); %floor(c*0.25);
iterations = 4; % for convergence
tampered_messages_per_test = 200;
tests = 1;

enable_guiding = false;
gamma_memory = 1;
threshold = 0;
filtering_rule = 'GWsTA';
propagation_rule = 'sum';
tampering_type = 'erase';

residual_memory = 0;
filtering_rule_first_iteration = false;
filtering_rule_last_iteration = false;

% Overlays
enable_overlays = true;
overlays_max = [0];
overlays_interpolation = {'uniform'};

% Plot tweaking
statstries = 5; % retry n times with different networks to average (and thus smooth) the results
smooth_factor = 1; % interpolate more points to get smoother curves. Set to 1 to avoid smoothing (and thus plot only the point of the real samples).
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
save_results = true; % save results to a file?

% == Launching the runs
D = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
E = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
TE = zeros(numel(M), numel(overlays_max)); % theoretical error rate depends on: Chi, l, c, erasures, enable_guiding and of course the density (theoretical or real) and thus on any parameter that changes the network (thus as the number of messages m to learn)
DError = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
DPredictedError = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
DWrong = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
DLost = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
DPropag = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
DGWTA = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));
LostEv = zeros(numel(M), numel(overlays_max)*numel(overlays_interpolation));

for t=1:statstries
    tperf = cputime(); % to show the total time elapsed later
    cnetwork = logical(sparse([]));
    thriftymessages = logical(sparse([]));
    for m=1:numel(M) % and for each value of m, we will do a run
        % Launch the run
        if m == 1
            [cnetwork, thriftymessages, density] = gbnn_learn('m', round(M(1, 1)*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                                                                        'enable_overlays', enable_overlays, ...
                                                                                                        'silent', silent);
        else % Optimization trick: instead of relearning the whole network, we will reuse the previous network and just add more messages, this allows to decrease the learning time exponentially, rendering it constant (at each learning, the network will learn the same amount of messages: eg: iteration 1 will learn 1E5 messages, iteration 2 will learn 1E5 messages and reuse 1E5, which will totalize as 2E5, etc...)
            [cnetwork, s2, density] = gbnn_learn('cnetwork', cnetwork, ...
                                                        'm', round((M(1, m)-M(1,m-1))*Mcoeff), 'miterator', miterator(1,m), 'l', l, 'c', c, 'Chi', Chi, ...
                                                        'enable_overlays', enable_overlays, ...
                                                        'silent', silent);
            thriftymessages = [thriftymessages ; s2]; % append new messages
        end

        counter = 1;
        for om=1:numel(overlays_max)
            for oi=1:numel(overlays_interpolation)
                [error_rate, theoretical_error_rate, test_stats] = gbnn_test('cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, ...
                                                                                      'erasures', erasures, 'iterations', iterations, 'tampered_messages_per_test', tampered_messages_per_test, 'tests', tests, ...
                                                                                      'enable_guiding', enable_guiding, 'gamma_memory', gamma_memory, 'threshold', threshold, 'propagation_rule', propagation_rule, 'filtering_rule', filtering_rule, 'tampering_type', tampering_type, ...
                                                                                      'residual_memory', residual_memory, 'filtering_rule_first_iteration', filtering_rule_first_iteration, 'filtering_rule_last_iteration', filtering_rule_last_iteration, ...
                                                                                      'enable_overlays', enable_overlays, 'overlays_max', overlays_max(om), 'overlays_interpolation', overlays_interpolation{oi}, ...
                                                                                      'silent', silent);

                % Store the results
                %colidx = counter+(size(D,2)/numel(enable_overlays))*(o-1);
                D(m,counter) = D(m,counter) + density;
                E(m,counter) = E(m,counter) + error_rate;
                TE(m, om) = theoretical_error_rate;
                if (tampered_messages_per_test*tests) > 0
                    DError(m, counter) = DError(m, counter) + test_stats.dtotalstats.tags_major_known_error/(tampered_messages_per_test*tests);
                    DPredictedError(m, counter) = DPredictedError(m, counter) + test_stats.dtotalstats.tags_error_predicted/(tampered_messages_per_test*tests);
                end
                if test_stats.dtotalstats.tags_error >0
                    % to get a normalized value between 0 and 1, divide either by: /(tampered_messages_per_test*tests); to get the normalized ratio of error among all test messages, or divide by /test_stats.dtotalstats.tags_error; to get normalized ratio of error among errored messages (but not correct ones).
                    DWrong(m, counter) = DWrong(m, counter) + test_stats.dtotalstats.tags_major_wrong_no_lost/(tampered_messages_per_test*tests); % /test_stats.dtotalstats.tags_error;
                    DLost(m, counter) = DLost(m, counter) + test_stats.dtotalstats.tags_major_init_lost/(tampered_messages_per_test*tests);
                    DPropag(m, counter) = DPropag(m, counter) + test_stats.dtotalstats.tags_major_propagfiltfail_only/(tampered_messages_per_test*tests);
                    DGWTA(m, counter) = DGWTA(m, counter) + test_stats.dtotalstats.tags_major_gwta_filtered_wrong_only/(tampered_messages_per_test*tests);
                end

                % Compute the fanal tag overwriting error
                fprintf('=> Computing the fanal tag overwriting error.\n'); aux.flushout(); % print total time elapsed
                net = cnetwork.primary.net;
                % First reassign tags uniformly if we have set a finite number of tags
                if overlays_max(om) > 0
                    maxa = max(nonzeros(net));
                    if maxa > overlays_max(om)
                        random_map = randi(overlays_max(om), maxa, 1);
                        net(net > 0) = random_map(nonzeros(net)); % faster!
                    end
                end

                % Compute the fanal tag overwriting rate, note that this depends solely on the structure of the network, ie: the learning step
                max_tag = M(m)*Mcoeff;
                init_lost_total = 0;
                for mi=1:max_tag
                    % Find if one of the fanals in the original clique lost its original tag on all of its edges (effectively losing this fanal for this clique)
                    fanals_idxs = find(thriftymessages'(:,mi));
                    init_edges = net(fanals_idxs, fanals_idxs);
                    init_lost = any(~any(ismember(init_edges, mi), 1));
                    % Find edges outside the clique but linked to the clique that got assigned to the same tag. If one such edge exists, then when we will decode we will inevitably also keep wrong fanals outside the clique.
                    init_conflict = 0;
                    if overlays_max(om) > 0
                        other_fanals_idxs = find(~thriftymessages'(:,mi));
                        other_edges = net(other_fanals_idxs, fanals_idxs); % other_edges = net(:, other_fanals_idxs);
                        init_conflict = any(any(ismember(other_edges, mi), 1));
                    end
                    % Compute the error
                    init_lost_total = init_lost_total + (init_lost || init_conflict);
                end
                % Compute the ratio
                LostEv(m, counter) = LostEv(m, counter) + (init_lost_total/max_tag);

                if ~silent; fprintf('-----------------------------\n\n'); end;

                counter = counter + 1;
            end
        end
    end
    aux.printcputime(cputime() - tperf, 'Total cpu time elapsed to do all runs: %G seconds.\n'); aux.flushout(); % print total time elapsed
end

% Normalizing errors rates by calculating the mean error for all tries
D = D ./ statstries;
E = E ./ statstries;
DError = DError ./ statstries;
DPredictedError = DPredictedError ./ statstries;
DWrong = DWrong ./ statstries;
DLost = DLost ./ statstries;
DPropag = DPropag ./ statstries;
DGWTA = DGWTA ./ statstries;
LostEv = LostEv ./ statstries;
fprintf('END of all tests!\n'); aux.flushout();

% Print densities values and error rates
%fprintf('Densities:\n'); disp(D);
%fprintf('Error rates:\n'); disp(E);
%fprintf('Theoretical error rates:\n'); disp(TE);
%aux.flushout();

% == Plotting

% -- First interpolate data points to get smoother curves
% Note: if smooth_factor == 1 then these commands won't change the data points nor add more.
nsamples = numel(M);
%isrowvect = @(v) isvector(v)&&size(v,1)==1;
M_interp = interp1(1:nsamples, M.', linspace(1, nsamples, nsamples*smooth_factor).', smooth_method); % interp1 third argument MUST be set as a column vector, so that the output stays in the same shape in case the second argument is a vector (this fixes a plot bug: if we plot a row vector, each point will be taken as one curve, instead of all the points. This is because the plot select points per column, thus we need to ensure that the plotted variables are at least a column vector and not row, when plotting only one curve).
D_interp = interp1(1:nsamples, D(:,1), linspace(1, nsamples, nsamples*smooth_factor).', smooth_method);
E_interp = interp1(D(:,1), E, D_interp, smooth_method);
TE_interp = interp1(D(:,1), TE, D_interp, smooth_method);
DError_interp = interp1(D(:,1), DError, D_interp, smooth_method);
DPredictedError_interp = interp1(D(:,1), DPredictedError, D_interp, smooth_method);
DWrong_interp = interp1(D(:,1), DWrong, D_interp, smooth_method);
DLost_interp = interp1(D(:,1), DLost, D_interp, smooth_method);
DPropag_interp = interp1(D(:,1), DPropag, D_interp, smooth_method);
DGWTA_interp = interp1(D(:,1), DGWTA, D_interp, smooth_method);
LostEv_interp = interp1(D(:,1), LostEv, D_interp, smooth_method);

% -- Save results to a file
if save_results
    blacklist_vars = {'cnetwork', 's2', 'thriftymessages', 'currentpath', 'currentscriptname', 'outfile', 'blacklist_vars'}; % vars to NOT save because they are really to huge (several MB or even GB)

    % Prepare filepath, filename and mkdir
    [currentpath, currentscriptname] = fileparts(mfilename('fullpath'));
    outfile = sprintf('%s/results/%s.mat', currentpath, currentscriptname);
    fprintf('Saving results into results/%s\n', currentscriptname);
    if ~isequal(exist('results', 'dir'),7)
        mkdir('results');
    end

    % Write data to file in MATLAB format
    %save(outfile, 'results'); % save ALL the workspace into a file
    aux.savex(outfile, blacklist_vars{:}); % save ALL the workspace into a file except for a few variables which are just too big
end

% Plot error rate with respect to the density (or number of messages stored) and a few other parameters
figure; hold on;
xlabel(sprintf('(Bottom) Density  -- (Top) Number of stored messages (M) x%.1E', Mcoeff));
ylabel('Retrieval Error Rate');
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)
for om=numel(overlays_max):-1:1
    for oi=1:numel(overlays_interpolation)
        colorcounter = om;
        if numel(overlays_interpolation) > 1; colorcounter = oi; end;
        coloridx = mod(colorcounter-1, numel(colorvec))+1; % change color if overlay or not

        lstyleidx = mod(counter-1, numel(linestylevec))+1; % change line style ...
        mstyleidx = mod(counter-1, numel(markerstylevec))+1; % and change marker style per plot

        lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        cur_plot = plot(D_interp, E_interp(:,end+1-counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx))); % plot one line
        set(cur_plot, plot_curves_params{:}); % additional plot style

        plot_title = sprintf('%s', filtering_rule);
        if enable_guiding
            plot_title = strcat(plot_title, sprintf(' - Guided'));
        else
            plot_title = strcat(plot_title, sprintf(' - Blind'));
        end
        if overlays_max(om) == 1
            plot_title = strcat(plot_title, sprintf(' - One/No tags'));
        elseif overlays_max(om) == 0
            plot_title = strcat(plot_title, sprintf(' - M tags'));
        else
            plot_title = strcat(plot_title, sprintf(' - %i tags', overlays_max(om)));
            plot_title = strcat(plot_title, sprintf(' (%s)', overlays_interpolation{oi}));
        end
        set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

        cc2 = 8;
        coloridx = mod(cc2-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        cur_plot = plot(D_interp, LostEv_interp(:,end+1-counter), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
        set(cur_plot, plot_curves_params{:}); % additional plot style
        set(cur_plot, 'DisplayName', strcat(plot_title, ' - lost evolution')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

        counter = counter + 1;
    end
end


% Plot theoretical error rates
if plot_theo
    %coloridx = mod(counter, numel(colorvec))+1;
    colornm = 'k';
    counter = 1;
    for om=numel(overlays_max):-1:1
        lstyleidx = mod(counter-1, numel(linestylevec))+1;
        mstyleidx = mod(counter-1, numel(markerstylevec))+1;

        lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
        cur_plot = plot(D_interp, TE_interp(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colornm)); % plot one line
        set(cur_plot, plot_curves_params{:}); % additional plot style

        plot_title = 'Theo. ';
        if enable_guiding
            plot_title = strcat(plot_title, sprintf(' - Guided'));
        else
            plot_title = strcat(plot_title, sprintf(' - Blind'));
        end
        if overlays_max(om) == 1
                plot_title = strcat(plot_title, sprintf(' - One/No tags'));
        elseif overlays_max(om) == 0
            plot_title = strcat(plot_title, sprintf(' - M tags'));
        else
            plot_title = strcat(plot_title, sprintf(' - %i tags', overlays_max(om)));
        end
        set(cur_plot, 'DisplayName', plot_title); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/

        counter = counter + 1;
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


% Plot tags error measures evolution
figure; hold on;
xlabel(sprintf('(Bottom) Density  -- (Top) Number of stored messages (M) x%.1E', Mcoeff));
ylabel('Error measures for tags');
ylim([0 1]);
om = numel(overlays_max); % set here the filtering rule that use disequilibrium
counter = 1; % useful to keep track inside the matrix E. This is guaranteed to be OK since we use the same order of for loops (so be careful, if you move the forloops here in plotting you must also move them the same way in the tests above!)

if overlays_max(om) == 1
    plot_title = 'One/No tags';
elseif overlays_max(om) == 0
    plot_title = 'M tags';
else
    plot_title = sprintf('%i tags', overlays_max(om));
end
%if concurrent_disequilibrium; plot_title = strcat(plot_title, ' + diseq'); end;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, E_interp(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(plot_title, ' - real error')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, DError_interp(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(plot_title, ' - tags error (known)')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, DPredictedError_interp(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(plot_title, ' - predicted error')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, DWrong_interp(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(plot_title, ' - wrong major tag')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, DLost_interp(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(plot_title, ' - lost the clique')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, DPropag_interp(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(plot_title, ' - propagation/filtering error')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, DGWTA_interp(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(plot_title, ' - gwta edges filter error')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

coloridx = mod(counter-1, numel(colorvec))+1; lstyleidx = mod(counter-1, numel(linestylevec))+1; mstyleidx = mod(counter-1, numel(markerstylevec))+1; lstyle = linestylevec(lstyleidx, 1); lstyle = lstyle{1}; % for MatLab, can't do that in one command...
cur_plot = plot(D_interp, LostEv_interp(:,om), sprintf('%s%s%s', lstyle, markerstylevec(mstyleidx), colorvec(coloridx)));
set(cur_plot, plot_curves_params{:}); % additional plot style
set(cur_plot, 'DisplayName', strcat(plot_title, ' - lost evolution')); % add the legend per plot, this is the best method, which also works with scatterplots and polar plots, see http://hattb.wordpress.com/2010/02/10/appending-legends-and-plots-in-matlab/
counter = counter + 1;

% Refresh plot with legends
legend(get(gca,'children'),get(get(gca,'children'),'DisplayName'), 'location', 'northwest'); % IMPORTANT: force refreshing to show the legend, else it won't show!
% Add secondary axis on the top of the figure to show the number of messages
%aux.add_2nd_xaxis(D(:,1), M, sprintf('x%.1E', Mcoeff), '%g', 0);
xlim([0 max(D(:,1))]); % adjust x axis zoom
% Adjust axis drawing style
set( gca(), plot_axis_params{:} );
% Adjust text style
set([gca; findall(gca, 'Type','text')], plot_text_params{:});


% The end!
