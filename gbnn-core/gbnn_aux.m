% #### Definition of some useful functions ####

% == Headers ==
% For Octave compatibility: we need the first function to have the same name as the filename.
function funs = gbnn_aux
    funs = importFunctions;
end

% For MatLab compatibility, we use a function to return other functions handlers as properties, this is a workaround since it cannot load multiple functions in one .m file (contrarywise to Octave using source())
function funs = importFunctions
    funs.shake=@shake; % the most important!
    funs.vertical_tile=@vertical_tile; % also important for concurrent_cliques to generate mixed messages
    funs.isOctave=@isOctave; % also important! for MatLab compatibility
    funs.getnargs=@getnargs; % to process named optional arguments
    funs.varspull = @varspull; % to load arguments into local namespace/workspace
    funs.fastmode=@fastmode;
    funs.kfastmode=@kfastmode;
    funs.colmode=@colmode;
    funs.nnzcolmode=@nnzcolmode;
    funs.flushout=@flushout; % to force refresh the stdout after printing in the console
    funs.printcputime=@printcputime;
    funs.printtime=@printtime;
    funs.editarg=@editarg;
    funs.delarg=@delarg;
    funs.addarg=@addarg;
    funs.rl_decode=@rl_decode;
    funs.interleave=@interleave;
    funs.interleaven=@interleaven;
    funs.add_2nd_xaxis=@add_2nd_xaxis;
    funs.savex=@savex;
    funs.binocdf=@binocdf;
end


% == Auxiliary functions ==

function [answer] = isOctave
% Returns true if this code is being executed by Octave.
% Returns false if this code is being executed by MATLAB, or any other MATLAB variant.
    persistent octaveVersionIsBuiltIn; % cache result
    if (isempty(octaveVersionIsBuiltIn))
        octaveVersionIsBuiltIn = (exist('OCTAVE_VERSION', 'builtin') == 5); % exist returns 5 to indicate a built-in function.
    end
    answer = octaveVersionIsBuiltIn; % isOctave cannot be persistent since this is the return variable, but another variable (octaveVersionIsBuiiltIn) can be persistent (it just must be a different variable than the returned variable).
end

function flushout
    if isOctave
        fflush(stdout);
        fflush(stderr);
    else
        drawnow('update');
    end
end

function printcputime(perf, sometext)
    if ~exist('sometext', 'var') || isempty(sometext)
        fprintf('Elapsed cpu time is %g seconds.\n', perf);
    else
        fprintf(sometext, perf);
    end
end

function printtime(perf, sometext)
    if ~exist('sometext', 'var') || isempty(sometext)
        fprintf('Elapsed time is %g seconds.\n', perf);
    else
        fprintf(sometext, perf);
    end
end

function [Y, I, J] = shake(X,dim)
    % SHAKE - Randomize a matrix along a specific dimension
    %   Y = SHAKE(X) randomizes the order of the elements in each column of the
    %   2D matrix. For N-D matrices it randomizes along the first non-singleton
    %   dimension.
    %
    %   SHAKE(X,DIM) randomizes along the dimension DIM.
    %
    %   [Y, I, J] = SHAKE(X) returns indices so that Y = X(I) and X = Y(J).
    %
    %   Example:
    %     A = [1 2 3 ; 4 5 6 ; 7 8 9 ; 10 11 12] ; % see <SLM> on the FEX ...
    %     Dim = 2 ;
    %     B = shake(A,Dim)  % ->, e.g.
    %      %  3     2     1
    %      %  6     4     5
    %      %  7     8     9
    %      % 11    10    12%
    %     C = sort(B,Dim) % -> equals A!
    %
    %     The function of SHAKE can be thought of as holding a matrix and shake
    %     in a particular direction (dimension), so that elements are getting
    %     shuffled within that direction only.
    %
    %   See also RAND, SORT, RANDPERM
    %   and RANDSWAP on the File Exchange

    % for Matlab R13
    % version 4.1 (may 2008)
    % (c) Jos van der Geest
    % email: jos@jasen.nl

    % History
    % Created: dec 2005
    % Revisions
    % 1.1 : changed the meaning of the DIM. Now DIM==1 works along the rows, preserving
    % columns, like in <sort>.
    % 2.0 (aug 2006) : randomize along any dimension
    % 2.1 (aug 2006) : output indices argument
    % 3.0 (oct 2006) : new & easier algorithm
    % 4.0 (dec 2006) : fixed major error in 3.0
    % 4.1 (may 2008) : fixed error for scalar input

    error(nargchk(1,2,nargin)) ;

    if nargin==1,
        dim = min(find(size(X)>1)) ;
    elseif (numel(dim) ~= 1) || (fix(dim) ~= dim) || (dim < 1),
        error('Shake:DimensionError','Dimension argument must be a positive integer scalar.') ;
    end

    % we are shaking the indices
    I = reshape(1:numel(X),size(X)) ;

    if numel(X) < 2 || dim > ndims(X) || size(X,dim) < 2,
        % in some cases, do nothing
    else
        % put the dimension of interest first
        [I,ndim] = shiftdim(I,dim-1) ;
        sz = size(I) ;
        % reshape it into a 2D matrix
        % we'll randomize along rows
        I = reshape(I,sz(1),[]) ;
        [~,ri] = sort(rand(size(I)),1) ;  % get new row indices
        ci = repmat([1:size(I,2)],size(I,1),1) ; % but keep old column indices
        I = I(sub2ind(size(I),ri,ci)) ; % retrieve values
        % restore the size and dimensions
        I = shiftdim(reshape(I,sz),ndim) ;
    end

    % re-index
    Y = X(I) ;

    if nargout==3,
        J = zeros(size(X)) ;
        J(I) = 1:numel(J) ;
    end

end

% Fastmode: mode implementation by Harold Bien that can return multiple results if multiple values are modes (while MatLab returns the smallest by default, and there's no way to change that).
function [y, n]=fastmode(x)
    % FASTMODE  Returns the most frequently occuring element
    %
    %   y = fastmode(x) returns the element in the vector 'x' that occurs the
    %   most number of times. If more than one element occurs at equal
    %   frequency, all elements with equal frequency are returned.
    %
    %   [y, n] = fastmode(x) does the same as above but also returns the
    %   frequency of the element(s) in 'n'.
    %
    %   Note that due to speed considerations, no error checking is performed
    %   on the input data. Matrices will be reduced to vectors via the colon
    %   (:) operator, NaN's are ignored.
    %
    %   Example
    %   % Generate a data set of values between 0 and 9
    %   >> data=fix(rand(1000,1).*9);
    %   % Get the most common value and its frequency
    %   >> [mode, n]=fastmode(data);
    %   % To confirm, run this simple script
    %   >> for i=1:9 disp(sprintf('Element %d: Frequency %d', i,
    %   length(data(data==i)))); end;
    %
    %   % Note if you give it only unique values, all values will be
    %   % returned with a frequency count of 1, i.e.
    %   >> [y, n]=fastmode([1:9]);
    %   % will result in y=[1:9] and n=1
    %
    %   See also MEAN, MEDIAN, STD.

    %   Copyright 2006 Harold Bien
    %   $Revision: 1.0 $    $Date: 2006/04/13 $

    % This code was evolved from mode.m by Michael Robbins and critical input
    % by others in MathWorks FileExchange

    % The data must be sorted in order for this algorithm to work
    sorted=sort(x(:));
    % Compute element-by-element difference. This will return 0 for
    % identical valued elements (since it is sorted) and non-zero for
    % different elements. We add a dummy element at the end in order to
    % pick up repeated elements at the end (make sure last element is not
    % equal to the next-to-last element). This value will never be used for
    % mode computations, so it's safe to add in.
    dist=diff([sorted; sorted(end)-1]);
    % This gives us unique values (presumably faster than unique())
    % [Profiler indicates a lot of time spent here, so we avoid it]
    %unique_vals=sorted(dist~=0);
    % Find locations of all non-zero elements and take the difference
    % which counts how many of each element is there. The first repeated
    % elements are dropped in the dist, so we add it back in as the first
    % non-zero index, i.e. the first non-zero index indicates that many
    % repeated first entries
    % Get non-zero entries
    idx=find(dist);
    num=[idx(1); diff(idx)];
    % Get the mode, including possible duplicates
    n=max(num);
    % Pull the values from the original sorted vector
    y=sorted(idx(num==n));
end

function [y, n]=kfastmode(x, k)
    % KFASTMODE  Returns the k most frequently occuring element

    % The data must be sorted in order for this algorithm to work
    sorted=sort(x(:));

    % Compute element-by-element difference.
    dist=diff([sorted; sorted(end)-1]);

    % Get non-zero entries
    idx=find(dist);
    num=[idx(1); diff(idx)];

    if k > 1 && numel(num) > 1
        % Get the k modes, including possible duplicates
        num_sorted = sort(num, 'descend');
        n = unique(num_sorted(1:k));
        % Pull the values from the original sorted vector
        y=sorted(idx(ismember(num,n)));
    else
        % Get the mode, including possible duplicates
        n=max(num);
        % Pull the values from the original sorted vector
        y=sorted(idx(num==n));
    end
end

function [modd,freq]=colmode(data)
    % This function calculates column-wise mode values of a matrix
    % If there are no modes in a column, it returns NaN
    % column mode values are stored in a cell array
    % [modd]=mode_calc(data)  gives only mode values
    % [modd,freq]=mode_calc(data)  gives mode values and their frequencies

    % IT'S NOT VERY PROFESSIONAL, BUT WORKS FINE
    % Ipek DEVECI KOCAKOC
    % ipek.deveciatdeu.edu.tr
    % June 2006

    [k,m]=size(data);
    data = sort(data)';
    for i=1:m
        dist=diff([data(i,:); data(i,end)-1]);
        idx=find(dist);
        num=[idx(1); diff(i,idx)];
        n=max(num);
        if n~=1
            y=sorted(idx(num==n));
        else
            y=NaN;
        end
        freq(:,i)=n;
        modd{i}=y;
    end
end

function [modd,freq] = nnzcolmode(data)
    % [modd,freq] = nnzcolmode(data)
    % This is a proxy function to use fastmode per-columns and without zeros
    % by Stephen Larroque
    % 7/72014

    m=size(data,2);
    data = sort(data);
    modd = cell(m, 1);
    freq = zeros(m, 1);
    parfor i=1:m
        x = nonzeros(data(:,i));
        if nnz(x) == 0
            modd{i} = NaN;
            freq(i) = NaN;
        else
            [modd{i}, freq(i)] = fastmode(nonzeros(data(:,i)));
        end
    end
end

function argStruct = getnargs(varargin, defaults, restrict_flag)
%GETNARGS Converts name/value pairs to a struct (this allows to process named optional arguments).
%
% ARGSTRUCT = GETNARGS(VARARGIN, DEFAULTS, restrict_flag) converts
% name/value pairs to a struct, with defaults.  The function expects an
% even number of arguments in VARARGIN, alternating NAME then VALUE.
% (Each NAME should be a valid variable name and is case sensitive.)
% Also VARARGIN should be a cell, and defaults should be a struct().
% Optionally: you can set restrict_flag to true if you want that only arguments names specified in defaults be allowed. Also, if restrict_flag = 2, arguments that aren't in the defaults will just be ignored.
% After calling this function, you can access your arguments using: argstruct.your_argument_name
%
% Examples:
%
% No defaults
% getnargs( {'foo', 123, 'bar', 'qwerty'} )
%
% With defaults
% getnargs( {'foo', 123, 'bar', 'qwerty'} , ...
%               struct('foo', 987, 'bar', magic(3)) )
%
% See also: inputParser
%
% Authors: Jonas, Richie Cotton and LRQ3000
%

    % Extract the arguments if it's inside a sub-struct (happens on Octave), because anyway it's impossible that the number of argument be 1 (you need at least a couple, thus two)
    if (numel(varargin) == 1)
        varargin = varargin{:};
    end

    % Sanity check: we need a multiple of couples, if we get an odd number of arguments then that's wrong (probably missing a value somewhere)
    nArgs = length(varargin);
    if rem(nArgs, 2) ~= 0
        error('NameValuePairToStruct:NotNameValuePairs', ...
            'Inputs were not name/value pairs');
    end

    % Sanity check: if defaults is not supplied, it's by default an empty struct
    if ~exist('defaults', 'var')
        defaults = struct;
    end
    if ~exist('restrict_flag', 'var')
        restrict_flag = false;
    end

    % Syntactic sugar: if defaults is also a cell instead of a struct, we convert it on-the-fly
    if iscell(defaults)
        defaults = struct(defaults{:});
    end

    optionNames = fieldnames(defaults); % extract all default arguments names (useful for restrict_flag)

    argStruct = defaults; % copy over the defaults: by default, all arguments will have the default value.After we will simply overwrite the defaults with the user specified values.
    for i = 1:2:nArgs % iterate over couples of argument/value
        varname = varargin{i};
        % check that the supplied name is a valid variable identifier (it does not check if the variable is allowed/declared in defaults, just that it's a possible variable name!)
        if ~isvarname(varname)
          error('NameValuePairToStruct:InvalidName', ...
             'A variable name was not valid: %s position %i', varname, i);
        % if options are restricted, check that the argument's name exists in the supplied defaults, else we throw an error. With this we can allow only a restricted range of arguments by specifying in the defaults.
        elseif restrict_flag && ~isempty(defaults) && ~any(strmatch(varname, optionNames))
            if restrict_flag ~= 2 % restrict_flag = 2 means that we just ignore this argument, else we show an error
                error('%s is not a recognized argument name', varname);
            end
        % else alright, we replace the default value for this argument with the user supplied one (or we create the variable if it wasn't in the defaults and there's no restrict_flag)
        else
            argStruct = setfield(argStruct, varname, varargin{i + 1});  %#ok<SFLD>
        end
    end

end

function varspull(s)
% Import variables in a structures into the local namespace/workspace
% eg: s = struct('foo', 1, 'bar', 'qwerty'); varspull(s); disp(foo); disp(bar);
% Will print: 1 and qwerty
%
%
% Author: Jason S
%
    for n = fieldnames(s)'
        name = n{1};
        value = s.(name);
        assignin('caller',name,value);
    end
end

function varargin = delarg(varname, varargin)
% varargin = delarg(varname, varargin)
% Removes an argument from varargin with name varname (varname must be either a string or a cell array of strings)

    % Extract the arguments if it's inside a sub-struct (happens on Octave), because anyway it's impossible that the number of argument be 1 (you need at least a couple, thus two)
    if (numel(varargin) == 1)
        varargin = varargin{:};
    end

    nArgs = length(varargin);
    for i = nArgs-1:-2:1 % iterate over couples of argument/value
        vname = varargin{i};
        if iscell(varname) && any(ismember(varname, vname))
            varargin(i:i+1) = [];
        elseif strcmp(vname, varname)
            varargin(i:i+1) = [];
            break;
        end
    end
end

function varargin = editarg(varname, varvalue, varargin)
% varargin = editarg(varname, varargin)
% Replaces an argument from varargin with name varname (varname must either be a string or a cell array of strings, same for varvalue) with the content varvalue

    % Extract the arguments if it's inside a sub-struct (happens on Octave), because anyway it's impossible that the number of argument be 1 (you need at least a couple, thus two)
    if (numel(varargin) == 1)
        varargin = varargin{:};
    end

    nArgs = length(varargin);
    for i = 1:2:nArgs % iterate over couples of argument/value
        vname = varargin{i};
        if iscell(varname) && any(ismember(varname, vname))
            idx = find(ismember(varname, vname));
            varargin{i+1} = varvalue{idx};
        elseif strcmp(vname, varname)
            varargin{i+1} = varvalue;
            break;
        end
    end
end

function varargin = addarg(varname, varvalue, varargin)
    varargin = {varargin ; varname ; varvalue};
end

function vec = rl_decode(len,val)
% vectorized run-length decoder
% from rude on FEX by us: http://www.mathworks.com/matlabcentral/fileexchange/6436-rude--a-pedestrian-run-length-decoder-encoder
    lx=and(len>0, ~(len==inf));
    if	~any(lx)
        vec=[];
        return;
    end
    if	numel(len) ~= numel(val)
        error(...
        sprintf(['rl-decoder: length mismatch\n',...
             'len = %-1d\n',...
             'val = %-1d'],...
              numel(len),numel(val)));
    end
    len=len(lx);
    val=val(lx);
    val=val(:).';
    len=len(:);
    lc=cumsum(len);
    lx=zeros(1,lc(end));
    lx([1;lc(1:end-1)+1])=1;
    lc=cumsum(lx);
    vec=val(lc);
end

function C = interleave(A, B, dimmode)
% Concatenate two matrices by interleaving them, either by row (dimmode == 1) or by column (dimmode == 2)
% Thank's to Peter Yu http://www.peteryu.ca/tutorials/matlab/interleave_matrices

    if size(A) ~= size(B)
        error('Size of the two supplied matrices does not match.')
        return;
    end
    if ~exist('dimmode', 'var') || isempty(dimmode)
        dimmode = 1
    end
    % Interleave by row
    if dimmode == 1
        C = reshape([A(:) B(:)]', 2*size(A,1), []);
    % Interleave by column
    else
        A = A';
        B = B';
        C = reshape([A(:) B(:)]', 2*size(A,1), [])';
    end
end

function C = interleaven(dimmode, varargin)
% Concatenate n matrices by interleaving them, either by row (dimmode == 1) or by column (dimmode == 2)
% Thank's to Peter Yu http://www.peteryu.ca/tutorials/matlab/interleave_matrices

    if ~exist('dimmode', 'var') || isempty(dimmode)
        dimmode = 1
    end

    % Interleave by row
    if dimmode == 1
        C = reshape(horzcat(varargin{:}), numel(varargin{1}), []); % we want to concatenate all vectorized versions of the matrices, same as cellfun(@(a) a(:), varargin, 'UniformOutput', false)
        C = reshape(C', numel(varargin)*size(varargin{1},1), []);
    % Interleave by column
    else
        varargin = cellfun(@transpose, varargin, 'UniformOutput', false);
        C = reshape(horzcat(varargin{:}), numel(varargin{1}), []);
        C = reshape(C', numel(varargin)*size(varargin{1},1), [])';
    end
end

function B = vertical_tile(A, ncols)
% B = vertical_tile(A, ncols)
% Vertically tile a matrix's columns by grouping ncols together and put the other columns below in the same fashion.
% Note: this is a loop-based solution because Octave doesn't yet support N-D matrices, but if that was possible, the vectorized solution would then be: reshape(permute(reshape(A,size(A,1),ncols,[]),[1 3 2]),[],ncols);

    B = [];
    if issparse(A)
        B = sparse(B);
    end

    for i=1:ncols
        B = [B A(:, i:ncols:end)];
    end
    B = reshape(B, [], ncols);
end



% == Plotting auxiliary functions ==

function add_2nd_xaxis(X, X2, X2_legend, num2str_format, text_rotation, text_params)
% Plot a second x axis at the top of the figure
% Usage: add_2nd_xaxis(X, X2, X2_legend, num2str_format, text_rotation, text_params)
% X = first X axis values
% X2 = values of the second X axis (must be of same size as X)
% X2_legend = an optional legend for the second X axis (will be placed at the top right)
% num2str_format = format to print the X2 values
% text_rotation = rotation of the X2 values (0 = horizontal, else specified in degrees from 0 to 360)
% text_params = other text parameters for the X2 values (as a cell array)
%


    if ~exist('num2str_format', 'var'); num2str_format = '%g'; end;
    if ~exist('text_rotation', 'var'); text_rotation = 0; end;
    if ~exist('text_params', 'var'); text_params = {}; end;

    %density_labels = cellfun(@(x) num2str(x, '%1.1e'), num2cell(D(:,1)), 'UniformOutput', false); % convert to a cell array (necessary to be passed to text()) + convert to a better numerical format %.0E
    messages_labels = cellfun(@(x) num2str(x, num2str_format), num2cell(X2), 'UniformOutput', false); % convert to a cell array (necessary to be passed to text()) + convert to a better numerical format %.0E
    xoffset_fix = (max(xlim)/100); % offset to the left because on the plot there's a glitch (as of Octave 3.8.1) which offsets a bit to the right...
    if (text_rotation == 0)
        yoffset_fix = ((max(ylim)-min(ylim))/20); % same for vertically, there is a small offset
    else
        yoffset_fix = ((max(ylim)-min(ylim))/40); % same for vertically, there is a small offset
    end
    if (exist('OCTAVE_VERSION', 'builtin') == 5) && strcmpi(graphics_toolkit(), 'gnuplot') % for GNUPLOT on Octave, we have to tweak a bit differently
        yoffset_fix = yoffset_fix * 1.5;
    end
    cur_text = text(X-xoffset_fix, ones(numel(X2), 1)*max(ylim)+yoffset_fix, messages_labels, 'Rotation', text_rotation, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left'); % draw the secondary axis as a simple text
    if ~isempty(text_params); set(cur_text, text_params{:}); end;
    if exist('X2_legend', 'var')
        legend_text = text(max(X) * 1.02, max(ylim)+yoffset_fix, X2_legend, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left'); % add the coefficient for the messages numbers
        if ~isempty(text_params); set(legend_text, text_params{:}); end;
    end

end % endfunction



% == System auxiliary functions ==

function savex(varargin)
% % SAVEX
% %
% % save all variables that are present in the workspace EXCEPT what you
% % specify explicitly (by including the variable names or by using regular
% % expressions).
% %
% % This is an alternative for Octave to the -regexp option in save() of MatLab, which allows to do: save(outfile, '-regexp', '^(?!(', strjoin(blacklist_vars, '|'), ')$).'); with blacklist_vars = {'var1', 'var2'};
% %
% % Author         : J.H. Spaaks
% % Date           : April 2009
% % MATLAB version : R2006a on Windows NT 6.0 32-bit
% %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% %
% % test1a = 0;
% % test2a = 2;
% % test3a = 4;
% % test4a = 6;
% % test5a = 8;
% %
% % test1b = 1;
% % test2b = 3;
% % test3b = 5;
% % test4b = 7;
% % test5b = 9;
% %
% % % This example saves all variables that are present in the workspace except
% % % 'test2a' and 'test5b'. 'test3' is ignored since there is no variable by
% % % that name:
% % savex('save-by-varname.mat','test2a','test3','test5b')
% %
% % % This example saves all variables that are present in the workspace except
% % % 'test4a', 'test4b' and 'test2b':
% % savex('save-by-regexp.mat','-regexp','test4[ab]','t[aeiou]st2[b-z]')
% %
% % % This example saves all variables that are present in the workspace except
% % % those formatted as Octave system variables, such as '__nargin__':
% % savex('no-octave-system-vars.mat','-regexp','^__+.+__$')
% %
% % % This example saves all variables that are present in the workspace except
% % % those which are specified using regular expressions, saving in ascii
% % % format. Supported options are the same as for SAVE.
% % savex('save-with-options.txt','-regexp','test4[ab]',...
% % 't[aeiou]st2[b-z]','-ascii')
% %
% %
% %
% % % clear
% % % load('save-by-varname.mat')
% % %
% % % clear
% % % load('save-by-regexp.mat')
% % %
% % % clear
% % % load('no-octave-system-vars.mat')
% % %
% % % clear
% % % load('save-with-options.txt','-ascii')
% % %


    varList = evalin('caller','who');
    saveVarList = {};

    if ismember(nargin,[0,1])
        % save all variables
        saveVarList = varList
        for u = 1:numel(saveVarList)
            eval([saveVarList{u},' = evalin(',char(39),'caller',char(39),',',char(39),saveVarList{u},char(39),');'])
        end
        save('matlab.mat',varList{:})

    elseif strcmp(varargin{2},'-regexp')
        % save everything except the variables that match the regular expression

        optionsList = {};
        inputVars ={};
        for k=3:numel(varargin)
            if strcmp(varargin{k}(1),'-')
                optionsList{1,end+1} = varargin{k};
            else
                inputVars{1,end+1} = varargin{k};
            end
        end


        for k=1:numel(varList)

            matchCell = regexp(varList{k},inputVars,'ONCE');

            matchBool = repmat(false,size(matchCell));
            for m=1:numel(matchCell)
                if ~isempty(matchCell{m})
                    matchBool(m) = true;
                end
            end
            if ~any(matchBool)
                saveVarList{end+1} = varList{k};
            end
        end

        for u = 1:numel(saveVarList)
            eval([saveVarList{u},' = evalin(',char(39),'caller',char(39),',',char(39),saveVarList{u},char(39),');'])
        end

        save(varargin{1},saveVarList{:},optionsList{:})



    elseif ischar(varargin{1})
        % save everything except the variables that the user defined in
        % varargin{2:end}
        optionsList = {};
        inputVars = {};
        for k=2:numel(varargin)
            if strcmp(varargin{k}(1),'-')
                optionsList{1,end+1} = varargin{k};
            else
                inputVars{1,end+1} = varargin{k};
            end
        end

        for k=1:numel(varList)

            if ~ismember(varList{k},inputVars)

                saveVarList{end+1} = varList{k};

            end

        end

        for u = 1:numel(saveVarList)
            eval([saveVarList{u},' = evalin(',char(39),'caller',char(39),',',char(39),saveVarList{u},char(39),');'])
        end

        save(varargin{1},saveVarList{:},optionsList{:})

    else
        error('Unknown function usage.')
    end
end %endfunction



% == Computation func ==

function cdf = binocdf (x, n, p)
%% Copyright (C) 2012 Rik Wehbring
%% Copyright (C) 1995-2013 Kurt Hornik
%%
%% This file is part of Octave.
%% Copied into the project to provide compatibility for MatLab users without requiring the STATISTICS toolbox.
%%
%% Octave is free software; you can redistribute it and/or modify it
%% under the terms of the GNU General Public License as published by
%% the Free Software Foundation; either version 3 of the License, or (at
%% your option) any later version.
%%
%% Octave is distributed in the hope that it will be useful, but
%% WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%% General Public License for more details.
%%
%% You should have received a copy of the GNU General Public License
%% along with Octave; see the file COPYING.  If not, see
%% <http://www.gnu.org/licenses/>.

%% -*- texinfo -*-
%% @deftypefn {Function File} {} binocdf (@var{x}, @var{n}, @var{p})
%% For each element of @var{x}, compute the cumulative distribution function
%% (CDF) at @var{x} of the binomial distribution with parameters @var{n} and
%% @var{p}, where @var{n} is the number of trials and @var{p} is the
%% probability of success.
%% @end deftypefn

%% Author: KH <Kurt.Hornik@wu-wien.ac.at>
%% Description: CDF of the binomial distribution

  if (nargin ~= 3)
    print_usage ();
  end

  if (~isscalar (n) || ~isscalar (p))
    [retval, x, n, p] = common_size (x, n, p);
    if (retval > 0)
      error ('binocdf: X, N, and P must be of common size or scalars');
    end
  end

  if (~isreal (x) || ~isreal (n) || ~isreal (p))
    error ('binocdf: X, N, and P must not be complex');
  end

  if (isa (x, 'single') || isa (n, 'single') || isa (p, 'single'));
    cdf = zeros (size (x), 'single');
  else
    cdf = zeros (size (x));
  end

  k = isnan (x) | ~(n >= 0) | (n ~= fix (n)) | ~(p >= 0) | ~(p <= 1);
  cdf(k) = NaN;

  k = (x >= n) & (n >= 0) & (n == fix (n) & (p >= 0) & (p <= 1));
  cdf(k) = 1;

  k = (x >= 0) & (x < n) & (n == fix (n)) & (p >= 0) & (p <= 1);
  tmp = floor (x(k));
  if (isscalar (n) && isscalar (p))
    cdf(k) = betainc (1 - p, n - tmp, tmp + 1);
  else
    cdf(k) = betainc (1 - p(k), n(k) - tmp, tmp + 1);
  end

end % endfunction

