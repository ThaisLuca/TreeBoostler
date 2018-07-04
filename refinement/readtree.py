import re

text = '''% FOR advisedby(A, B):
%   if ( hasposition(B, C), student(A) )
%   then if ( publication(D, B), publication(D, A) )
%   | then return 0.8581489350995117;  // std dev = 2.46e-07, 41.000 (wgt'ed) examples reached here.  /* #pos=41 */
%   | else if ( publication(E, A), publication(E, F), professor(F) )
%   | | then if ( ta(G, A), publication(H, B) )
%   | | | then return 0.8581489350995121;  // std dev = 2.11e-08, 3.000 (wgt'ed) examples reached here.  /* #pos=3 */
%   | | | else if ( publication(I, B), tempadvisedby(J, F), publication(K, J) )
%   | | | | then return -0.14185106490048777;  // std dev = 0.000, 5.000 (wgt'ed) examples reached here.  /* #neg=5 */
%   | | | | else if ( taughtby(L, B), ta(L, M), publication(N, M) )
%   | | | | | then return 0.0803711573217344;  // std dev = 0.416, 9.000 (wgt'ed) examples reached here.  /* #neg=7 #pos=2 */
%   | | | | | else if ( hasposition(F, C) )
%   | | | | | | then return 0.8581489350995122;  // std dev = 0.000, 4.000 (wgt'ed) examples reached here.  /* #pos=4 */
%   | | | | | | else return 0.10814893509951219;  // std dev = 0.866, 4.000 (wgt'ed) examples reached here.  /* #neg=3 #pos=1 */
%   | | else if ( ta(P, A), taughtby(P, Q) )
%   | | | then return 0.49644680743993685;  // std dev = 0.480, 47.000 (wgt'ed) examples reached here.  /* #neg=17 #pos=30 */
%   | | | else return 0.6959867729373493;  // std dev = 0.369, 37.000 (wgt'ed) examples reached here.  /* #neg=6 #pos=31 */
%   else return -0.1365879070057512;  // std dev = 0.072, 190.000 (wgt'ed) examples reached here.  /* #neg=189 #pos=1 */'''

# Code responsible for reading tree and its standard deviations
# and generating a refinement.txt
lines = text.split('\n')
current = []
stack = []
target = None
nodes = {}
stdDevs = {}

for line in lines:
  print(stack)
  if not target:
    match = re.match('\s*\%\s*FOR\s*(\w+\([\w,\s]*\)):', line)
    if match:
      target = match.group(1)
  match = re.match('.*if\s*\(\s*([\w\(\),\s]*)\s*\).*', line)
  if match:
    nodes[','.join(current)] = match.group(1).strip()
    clause = match.group(1)
    stack.append(current+['false'])
    current.append('true')
  match = re.match('.*then return [\d.-]*;\s*\/\/\s*std dev\s*=\s*([\d.\-e]*).*', line)
  if match:
    stdDevs[','.join(current)] = float(match.group(1))
    if len(stack):
      current = stack.pop()
  match = re.match('.*else return [\d.-]*;\s*\/\/\s*std dev\s*=\s*([\d.\-e]*).*', line)
  if match:
    stdDevs[','.join(current)] = float(match.group(1))
    if len(stack):
      current = stack.pop()