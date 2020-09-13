% bt_lsearch file
function a = bt_lsearch(x,d,fname,gname,p1,p2)
rho = 0.1;
gma = 0.5;
x = x(:);
d = d(:);
a = 1;
xw = x + a*d;
ps ='';
if nargin == 5
   if ischar(p1)
      eval([p1 ';']);
   else
      ps = ',p1';
   end
end
if nargin == 6
   if ischar(p1)
      eval([p1 ';']);
   else
      ps = ',p1';
   end
   if ischar(p2)
      eval([p2 ';']);
   else
      ps = ',p1,p2';
   end
end
eval(['f0 = ' fname '(x' ps ');']);
eval(['g0 = ' gname '(x' ps ');']);
eval(['f1 = ' fname '(xw' ps ');']);
f2 = f0 + rho*a*g0'*d;
er = f1 - f2;
while er > 0
     a = gma*a;
     xw = x + a*d;
     eval(['f1 = ' fname '(xw' ps ');']);
     f2 = f0 + rho*a*g0'*d;
     er = f1 - f2;
end
if a < 1e-5
   a = min([1e-5, 0.1/norm(d)]); 
end 
