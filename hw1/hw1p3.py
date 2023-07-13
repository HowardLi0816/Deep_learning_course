# -*- coding: utf-8 -*-
from func import f
import sys
import random
'''
def secant_iter(a, b, citer):
    if (isinstance(a, int) or isinstance(a, float)) and (isinstance(b, int) or isinstance(b, float)):
        if a < b:
            if f(a)*f(b) < 0:
                a_ran = random.random()
                b_ran = random.random()
                x0 = a + a_ran * (b - a)
                #print("x0", x0)
                x1 = a + b_ran * (b - a)
                #print("x1", x1)
                ori_x0 = x0
                N = 0
                while abs(x1-x0) >= citer:
                    x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
                    N += 1
                    ori_x0 = x0
                    x0 = x1
                    #print("x0", x0)
                    x1 = x2
                    #print("x1", x1)
                #stdout = [N, '\n', ori_x0, '\n', x0, '\n', x1]
                #print(stdout)
                #print(N, '\n', ori_x0, '\n', x0, '\n', x1)
                #print(str(N))
                #print(str(ori_x0))
                #print(str(x0))
                #print(str(x1))
                sys.stdout.write(str(N)+'\n'+str(ori_x0)+'\n'+str(x0)+'\n'+str(x1))
            else:
                #sys.stderr.write("Range Error: According to the Bolzano's Theorem, f(a)*f(b)<0! Please enter again!")
                sys.stderr.write('range error')
        else:
            #sys.stderr.write("Range Error: The bound [a, b] must satisfy a<b! Please enter again")
            sys.stderr.write('range error')
    else:
        #sys.stderr.write("Range Error: The bound a and b must be numeric! Please enter again!")
        sys.stderr.write('range error')
    #return N, ori_x0, x0, x1
'''

def secant_iter(a, b, citer):
    #print(a)
    #print(b)
    #print(isinstance(a, int), isinstance(a, float), isinstance(b, int), isinstance(b, float))
    #print(isinstance(a, float))
    if not isinstance(a, float) or not isinstance(b, float):
        sys.stderr.write('range error')
        return
    if a >= b:
        sys.stderr.write('range error')
        return
    if f(a) * f(b) >= 0:
        sys.stderr.write('range error')
        return


    '''
    a_ran = random.random()
    b_ran = random.random()
    x0 = a + a_ran * (b - a)
    #print("x0", x0)
    x1 = a + b_ran * (b - a)
    #print("x1", x1)
    '''
    x0 = a
    x1 = b
    ori_x0 = x0
    N = 0
    while abs(x1-x0) >= citer:
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        N += 1
        ori_x0 = x0
        x0 = x1
        #print("x0", x0)
        x1 = x2
        #print("x1", x1)
    #stdout = [N, '\n', ori_x0, '\n', x0, '\n', x1]
    #print(stdout)
    #print(N, '\n', ori_x0, '\n', x0, '\n', x1)
    #print(str(N))
    #print(str(ori_x0))
    #print(str(x0))
    #print(str(x1))
    sys.stdout.write(str(N)+'\n'+str(ori_x0)+'\n'+str(x0)+'\n'+str(x1)+'\n')
    #print(ori_x0)
    return



if __name__ == "__main__":
    a = float(sys.argv[1])
    #print(type(sys.argv[1]))
    #print(a)
    #print(isinstance(a, float))
    b = float(sys.argv[2])
    secant_iter(a, b, 1e-10)
    
    '''
    N, xN_2, xN_1, xN = secant_iter(a, b, 1e-10)
    print(N)
    print(xN_2)
    print(xN_1)
    print(xN)
    '''