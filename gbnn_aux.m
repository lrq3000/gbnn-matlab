% #### Definition of some useful functions ####

% == Headers ==
% For Octave compatibility: we need the first function to have the same name as the filename.
function funs = gbnn_aux
    funs = importFunctions;
end

% For MatLab compatibility, we use a function to return other functions handlers as properties, this is a workaround since it cannot load multiple functions in one .m file (contrarywise to Octave using source())
function funs = importFunctions
    funs.shake=@shake; % the most important!
    funs.fastmode=@fastmode;
    funs.isOctave=@isOctave;
    funs.flushout=@flushout;
    funs.printcputime=@printcputime;
    funs.printtime=@printtime;
end


% == Auxiliary functions ==

function [answer] = isOctave
% Returns true if this code is being executed by Octave.
% Returns false if this code is being executed by MATLAB, or any other MATLAB variant.
    persistent octaveVersionIsBuiltIn; % cache result
    if (isempty(octaveVersionIsBuiltIn))
        octaveVersionIsBuiltIn = (exist('OCTAVE_VERSION', 'builtin') == 5); % exist returns 5 to indicate a built-in function.
    end
    answer = octaveVersionIsBuiltIn; % isOctave cannot be persistent since this is the return variable
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
