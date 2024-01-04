import os
import numpy as np

top_path=".\\testruns"
dirnm=top_path

cmd = 'mkdir '+dirnm

if not os.path.isdir(top_path):
    print('toppath not here')
    print(cmd)
    os.system(cmd)

dirnm=top_path+'\\TestDir'
cmd = 'mkdir '+dirnm

if not os.path.isdir(dirnm):
    print('dir not here')
    print(cmd)
    os.system(cmd)

fname = dirnm + '\\test.txt'
x = np.arange(9).reshape(3,3)
np.savetxt(fname,x)

fwrite = open(fname, 'w')
fwrite.write('WROTE')
fwrite.close()


print('DONE')