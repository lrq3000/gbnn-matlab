% Test unit for the gbnn network matlab/octave package
% great to check compatibility between matlab/octave!
% Note: this only tests that the network works with various configurations, but it does not check the results!

% Clear things up
clear all; % don't forget to clear all; before, else some variables or sourcecode change may not be refreshed and the code you will run is the one from the cache, not the latest edition you did!
close all;

% Importing auxiliary functions
aux = gbnn_aux; % works with both MatLab and Octave

% Vars config, tweak the stuff here
m = 2; %1E4;
c = 12;
l = 32;
Chi = 64;
erasures = 3;
tampered_messages_per_test = 5;
silent = true;

% Setup the test cases
% Format: a big cell array, containing one cell array per test case, each test case containing one string for the title and two cell arrays: one for the learning parameters, and one for the testing parameters
test_cases = { ...
                            { ...
                                ... % 1st test case: concurrent + ml
                                'filtering rule: ML + concurrent',
                                {},
                                {'concurrent_cliques', 2, 'filtering_rule', 'ML'}
                                ... % 2nd test case
                                'filtering rule: GWsTA'
                                {},
                                {'concurrent_cliques', 1, 'filtering_rule', 'GWsTA'}
                            }
                            
                        };

% == Launching the runs
for i=1:numel(test_cases)
    all_args = test_cases{i};
    title_arg = all_args{1};
    learn_args = all_args{2};
    test_args = all_args{3};
    printf('=================\n== Test case %i/%i: %s ==\n=================\n', i, numel(test_cases), title_arg); aux.flushout();
    tperf = cputime();
    [cnetwork, thriftymessages, density] = gbnn_learn('silent', silent, 'm', m, 'l', l, 'c', c, 'Chi', Chi, learn_args{:});
    error_rate = gbnn_test('silent', silent, 'cnetwork', cnetwork, 'thriftymessagestest', thriftymessages, 'tampered_messages_per_test', tampered_messages_per_test, 'erasures', erasures, test_args{:});
    aux.printcputime(cputime() - tperf, sprintf('Finished test case %i%s', i, 'total cpu time elapsed to do everything: %g seconds.\n\n')); aux.flushout(); % print total time elapsed
end

printf('If you have got no error thus far, then alright! All test cases passed!\n');

% The end!
