import numpy as np
import csv
import argparse

def Get_tiles(opmat, pod, SAsize):
    opmat_size = opmat
    SA = SAsize
    Pods = pod

    #compute
    Tiles = int(np.ceil(opmat_size/SA))
    Used_pod = []
    for i in range(Pods):
        temp = np.ceil(Tiles/(i+1))
        Used_pod.append(temp)
    Used_pod = np.argmin(Used_pod)+1 #runtime은 가장 짧으면서 가능한 적은 pod 쓰도록.

    Tiles_each_pod = []
    Tiles_each_pod_acc = []
    for j in range (Used_pod):
        Tiles_each_pod.append(0)
        Tiles_each_pod_acc.append(0)

    for k in range(Tiles):
        if (k==Tiles-1)&(opmat_size % SA != 0):
            Tiles_each_pod[k%Used_pod] = Tiles_each_pod[k%Used_pod]+(opmat_size % SA)
        else:
            Tiles_each_pod[k%Used_pod] = Tiles_each_pod[k%Used_pod]+SA
    #print(Tiles_each_pod)

    for l in range(len(Tiles_each_pod)):
        Tiles_each_pod_acc[l]=np.sum(Tiles_each_pod[0:(l+1)])

    return Tiles_each_pod

def Get_Util(SA_row, SA_col, SR, SC):
    #Fold 수 계산
    FR=int(np.ceil(SR/SA_row))
    FC=int(np.ceil(SC/SA_col))
    
    Util_row = np.zeros(FR)
    for i in range(FR):
        if (i == (FR-1)) & ((SR % SA_row) != 0):
            Util_row[i] = SR % SA_row
        else:
            Util_row[i] = SA_row
    
    Util_col = np.zeros(FC)
    for i in range(FC):
        if (i == (FC-1)) & ((SC % SA_col) != 0):
            Util_col[i] = SC % SA_col
        else:
            Util_col[i] = SA_col

    Utilization = np.zeros([FR, FC])
    for i in range(FR):
        for j in range(FC):
            Utilization[i][j] = (Util_row[i] * Util_col[j]) / (SA_row * SA_col)
    Utilization = np.average(Utilization)

    return Utilization

######################################################################

parser = argparse.ArgumentParser(description='')
parser.add_argument('topo', help='Set topology file')
args = parser.parse_args()
toponame= args.topo #topology 이름 (e.g.,alexnet)
#toponame= 'Googlenet' #topology 이름 (e.g.,alexnet)
if toponame[0:4]=="BERT":
    topofile = './topologies/TFs_mnk/'+toponame+'.csv'
else:
    topofile = './topologies/conv_nets/'+toponame+'.csv'
csvtopo=open(topofile,'r',encoding='cp949')

dataflow='os'

E_MAC = 0.4
E_BUF = 2.7

MAC_arr=[]


# 4 간격으로 진행 ##########################################
temp_arr=[]
for i in range(32):
        temp_arr.append((i+1)*4)

for i in temp_arr:
    for j in temp_arr:
        # if (i in temp_arr_old)&(j in temp_arr_old):
        #     continue
        SAdim = []
        SAdim.append(i)
        SAdim.append(j)
        MAC_arr.append(SAdim)
#############################################################

repname='./TPWresults/'+toponame+'.csv'
report_file = open(repname,'r')
rep_reader = csv.reader(report_file)
rep_header = next(rep_reader)

over_repname = './TPWresults/' + toponame + '_result.csv'
overall_report_file = open(over_repname, 'w', newline='')
rep_writer = csv.writer(overall_report_file)
# rep_writer.writerow(["SA_row", "SA_col", "Runtime", "DRAM access", "SRAM access", "PE util", "DRAM R", "DRAM W", "SRAM R", "SRAM W",\
#  "P. Throughput (GFLOPS/S)", "E. Throughput (GFLOPS/S)", "MAC Energy (pJ)", "BUF Energy (pJ)", "Total Energy (pJ)", "Power Consumption (mW)", "P.Throughput/Watt (TOPS/W)", "E.Throughput/Watt (TOPS/W)"])

rep_writer.writerow(["SA_row", "SA_col", "Runtime", "SRAM access", "PE util",\
     "P. Throughput (GFLOPS/S)", "E. Throughput (GFLOPS/S)", "MAC Energy (pJ)", "BUF Energy (pJ)", "Total Energy (pJ)", "Power Consumption (mW)", "P.Throughput/Watt (TOPS/W)", "E.Throughput/Watt (TOPS/W)"])

for rows in rep_reader:
    MACs_row=int(rows[0])
    MACs_col=int(rows[1])
    
    part_dims=[]
    part_dims=[[1,1]]

    Runtime_result=[]
    Util_result=[]
    pod_usage_result=[]

    if dataflow == 'os':
        for part_dim in part_dims:
            #print("### {}x{} partition ###".format(part_dim[0],part_dim[1]))
            OSruntime=[]
            OVERruntime=[]
            OVERutil=[]
            ROWUSAGE=[]
            COLUSAGE=[]
            PODUSAGE=[]

            csvtopo=open(topofile,'r',encoding='cp949')
            toporow=csv.reader(csvtopo)
            next(toporow)
            for topo in toporow:
                if toponame[0:4]=="BERT":
                    m=int(topo[1])
                    n=int(topo[2])
                    k=int(topo[3])

                    SA_row=MACs_row/part_dim[0]
                    SA_col=MACs_col/part_dim[1]

                    SR=m
                    SC=n
                    T=k
                else:
                    I_row=int(topo[1])
                    I_col=int(topo[2])
                    F_row=int(topo[3])
                    F_col=int(topo[4])
                    S=int(topo[7])
                    Chan=int(topo[5])
                    numFil=int(topo[6])

                    SA_row=MACs_row/part_dim[0]
                    SA_col=MACs_col/part_dim[1]

                    numOF=np.ceil((I_row-F_row+S)/S)*np.ceil((I_col-F_col+S)/S)
                    Wconv=F_row*F_col*Chan

                    SR=numOF
                    SC=numFil
                    T=Wconv

                #Tiling
                IFMAP_tiles = Get_tiles(SR, part_dim[0], SA_row)
                Filter_tiles = Get_tiles(SC, part_dim[1], SA_col)

                OSruntime=[]
                Util_arr=[]
                for ifmap in IFMAP_tiles:
                    for filters in Filter_tiles:
                        SR=ifmap
                        SC=filters
                        T=T
                        
                        Util = Get_Util(SA_row, SA_col, SR, SC) #pod별 util 구함
                        Util_arr.append(Util) #각 레이어에 대한 pod들의 utilization array

                        runt=(SA_row + SA_col + T-2)*np.ceil(SR/SA_row)*np.ceil(SC/SA_col) #논문 analytical model은 2*SA_row.
                        OSruntime.append(runt)
                Util_this_layer = np.average(Util_arr)

                OVERruntime.append(np.max(OSruntime)) #각 layer별 max runtime
                OVERutil.append(Util_this_layer)
            OVERutil=np.round(np.asarray(OVERutil)*100, 2)
            Util_full_layer = np.round(np.average(OVERutil), 2)

            Runtime_result.append(np.sum(OVERruntime))
            Util_result.append(Util_full_layer)


    #print('*Weighted Average PE Util*')
    WUTIL=[]
    for i in range(len(OVERruntime)):
        temp=(OVERruntime[i]/np.sum(OVERruntime))*OVERutil[i]
        WUTIL.append(temp)
    Wutil = np.round(np.sum(WUTIL),2)
    P_Throughput = 1 * 2 * MACs_row * MACs_col
    E_Throughput = P_Throughput * Wutil * 0.01
    MACenergy = MACs_row * MACs_col * float(rows[2]) * E_MAC
    BUFenergy = float(rows[3]) * E_BUF
    TOTALenergy = MACenergy + BUFenergy
    Pconsump = TOTALenergy / float(rows[2])
    P_TPW = P_Throughput / Pconsump
    E_TPW = E_Throughput / Pconsump

    # ["SA_row", "SA_col", "Runtime", "SRAM access", "PE util",\
    #  "P. Throughput (GFLOPS/S)", "E. Throughput (GFLOPS/S)", "MAC Energy (pJ)", "BUF Energy (pJ)", "Total Energy (pJ)", "Power Consumption (mW)", "P.Throughput/Watt (TOPS/W)", "E.Throughput/Watt (TOPS/W)"]
    result_this_row = [int(rows[0]), int(rows[1]), float(rows[2]), float(rows[3]), Wutil, P_Throughput, E_Throughput, MACenergy, BUFenergy, TOTALenergy, Pconsump, P_TPW, E_TPW]
    rep_writer.writerow(result_this_row)


csvtopo.close()
report_file.close()
overall_report_file.close()