import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser(description='')
parser.add_argument('topo', help='Set topology file')
args = parser.parse_args()
toponame= args.topo #topology 이름 (e.g.,alexnet)
topofile = './topologies/conv_nets/'+toponame+'.csv'
csvtopo=open(topofile,'r',encoding='cp949')

dataflow='ws'

MACs_row=128
MACs_col=128
pod_row=4
pod_col=4

Runtime_arr=[]

SA_row = int(MACs_row/pod_row)
SA_col = int(MACs_col/pod_col)

num_MAC = []

csvtopo=open(topofile,'r',encoding='cp949')
toporow=csv.reader(csvtopo)
next(toporow)
for topo in toporow:
    I_row=int(topo[1])
    I_col=int(topo[2])
    F_row=int(topo[3])
    F_col=int(topo[4])
    S=int(topo[7])
    Chan=int(topo[5])
    numFil=int(topo[6])

    numOF=np.ceil((I_row-F_row+S)/S)*np.ceil((I_col-F_col+S)/S)
    Wconv=F_row*F_col*Chan

    SR=Wconv
    SC=numFil
    T=numOF

    if_row = T
    if_col = SR
    fil_row = SR
    fil_col = SC
    #print(if_row)

    num_MAC_this_layer = T*SR*SC
    num_MAC.append(num_MAC_this_layer)

num_MAC = sum(num_MAC)
print('Number of MACs: {}'.format(num_MAC))