import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser(description='')
parser.add_argument('topo', help='Set topology file')
args = parser.parse_args()
toponame= args.topo #topology 이름 (e.g.,alexnet)
topofile = './topologies/TFs_mnk/'+toponame+'.csv'
csvtopo=open(topofile,'r',encoding='cp949')

num_MAC = []

csvtopo=open(topofile,'r',encoding='cp949')
toporow=csv.reader(csvtopo)
next(toporow)
for topo in toporow:
    M=int(topo[1])
    N=int(topo[2])
    K=int(topo[3])

    num_MAC_this_layer = M*N*K
    num_MAC.append(num_MAC_this_layer)

num_MAC = sum(num_MAC)
print('Number of MACs: {}'.format(num_MAC))