function [sse,func] = nakaRushton(c,par,data)
% fits dprime
% fits and calculates sse for a NakaRushton function
% rmax - asymptotic performance 
% c    - stimulus levels
% n    - slope
% c50  - contrast at which the observer achieves half the asymptotic
%        performance
% data - is well the data...
% if less than 4 arguments will be used for interpolation
% minimized using sse
% @author: antonio fernandez

n    = par(1);
c50  = par(2);   
rmax = par(3);
func = (rmax*c.^n)./(c.^n+c50^n);

% can weigh the error to help functions saturate
w = ones(size(c))'; %[1,1,1,1,1,1] for equal weights
if nargin == 3
	if length(c) ~= length(data)
		error('x and data do not have the same length.')
	end
	sse = sum(((data-func).*w').^2);  
else
	sse = NaN;
end

end