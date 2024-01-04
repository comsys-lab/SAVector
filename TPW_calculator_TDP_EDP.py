import numpy as np
import csv
import argparse

def get_PE_ratio(SA_row, SA_col, this_m, Tbuf, Tmac):
    base_rt = (SA_row + this_m + SA_col - 1)
    
    # Shared buffer access latency
    # In this eq, Tmac is tile operation cycles, not a single MAC cycle.
    # Assume that pods fetch next input tile immediately after processing the current tile is completed.
    # No perf drop when Tbuf = 1
    shared_buffer_delay = max(1.0,((Tbuf - 1 + Tmac) / SA_col))
    fold_rt = base_rt * shared_buffer_delay
    #
    this_active_PE = 100*(0.5*(SA_row + SA_col) + this_m - 1)/fold_rt

    
    # print("base_rt: {}\nRT increase due to the shbuf latency: {}\nRT increase due to the offmem latency: {}\n".format(base_rt, shared_buffer_delay, offchip_perf))
    
    return this_active_PE

parser = argparse.ArgumentParser(description='')
parser.add_argument('topo', help='Set topology file')
parser.add_argument('repname', help='Set report file to read')
parser.add_argument('repfile', help='Set report file to read')
parser.add_argument('SAsize', help='Set report file to read')
args = parser.parse_args()
toponame= args.topo #topology name (e.g.,alexnet)

if "BERT" in toponame:
    mnk_flag = True
else:
    mnk_flag = False

SA_dim = args.SAsize
SA_row, SA_col = map(int, SA_dim.split('x'))
MACs_per_pod = SA_row * SA_col # number of total MAC units


E_MAC = 0.48 # pJ
E_MAC_mul_add = 0.23
E_MAC_onlyprop = E_MAC-E_MAC_mul_add # pJ
E_MAC_static = 0.017 # pJ
E_BUF = 3.16 # pJ
# E_BUF_large = 8.06 # pJ
E_tinyBUF = 0.15 # pJ
E_offmem = 31.2 # pJ, HBM=31.2, DDR4=140, GDDR6=112
Clock_freq = 1000000000 # 1GHz

if SA_dim == "16x16":
    Peak_P = 0.57 # W
    E_BUF = 2.14 # pJ
elif SA_dim == "32x32":
    Peak_P = 1.11 # W
elif SA_dim == "64x64":
    Peak_P = 2.78 # W
    E_BUF = 4.69 # pJ

# Report file to open
repname = args.repfile
report_file = open(repname,'r')
rep_reader = csv.reader(report_file)
rep_header = next(rep_reader)

# Report file to write
repname=args.repname
over_repname = './TDP_calculated_SOSA/TPW_' + repname + '.csv'
overall_report_file = open(over_repname, 'w', newline='')
rep_writer = csv.writer(overall_report_file)

#Batch,Cutil,PeakT,EffT,MACP,SRAMP,OffP,P,EffTPW
# rep_writer.writerow(["Num pod", "PP", "Compute util", "P. Throughput (GFLOPS/S)", "E. Throughput (GFLOPS/S)",\
#      "MAC Power (mW)", "SRAM Power (mW)", "Offmem Power (mW)", "Power Consumption (mW)", "P.Throughput/Watt (TOPS/W)", "E.Throughput/Watt (TOPS/W)"])
rep_writer.writerow(["Num pod", "PP", "RT", "TOTAL_E", "EDP"])

for rows in rep_reader:
    RT=int(float(rows[0]))
    Offmem=int(float(rows[1]))
    SRAM=int(float(rows[2]))
    Offmem_R=int(float(rows[3]))
    Offmem_W=int(float(rows[4]))
    SRAM_R=int(float(rows[5]))
    SRAM_W=int(float(rows[6]))
    Comp_util=float(rows[7])
    PE_util=float(rows[8])
    Pod_util=float(rows[9])
    Batch=int(rows[10])
    Num_pod=int(rows[11])
    Pod_row=int(rows[12])
    Pod_col=int(rows[13])
    
    #
    RT=RT/1000000 # ms
    #
    this_m = 4096/Pod_row
    Tmac = int(SA_col) # Cycles per MAC operation, default=1
    Tbuf = int(2 + np.sqrt(Num_pod)*4)+1 # Shared buffer access latency
    # Tbuf = 1 # Shared buffer access latency
    PE_ratio=get_PE_ratio(SA_row, SA_col, this_m, Tbuf, Tmac)
    #
    MACs_row = SA_row * Pod_row
    MACs_col = SA_col * Pod_col
    #
    MAC_E = (E_MAC_mul_add * MACs_row * MACs_col) * (RT * (Comp_util * 0.01)) + (E_MAC_onlyprop * MACs_row * MACs_col) * (Pod_util * 0.01) * (RT * (PE_ratio * 0.01)) + ((E_MAC_static * MACs_row * MACs_col) * (Pod_util * 0.01) * RT)
    MAC_E = np.round(MAC_E/np.power(10,9),2)
    # SRAM_E = (E_BUF+E_tinyBUF) * SRAM
    SRAM_E = (E_BUF) * SRAM
    SRAM_E = np.round(SRAM_E/np.power(10,9),2)
    Offmem_E = Offmem * E_offmem
    Offmem_E = np.round(Offmem_E/np.power(10,9),2)
    # Offmem_E = 0
    
    Total_E = np.round(MAC_E + SRAM_E + Offmem_E,2)
    
    EDP = np.round(Total_E * RT, 1)
    
    
    
    # P_Throughput = (2 * MACs_per_pod * Num_pod) # GFLOPS
    # E_Throughput = P_Throughput * Comp_util * 0.01 # GFLOPS
    # MAC_power = ((E_MAC * MACs_per_pod * Num_pod) + (E_MAC_static * MACs_per_pod * Num_pod)) * (Pod_util * 0.01) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
    # if 'Scale_up' in repname:
    #     SRAM_power = ((SRAM * (E_BUF_large)) / RT) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
    # else:
    #     SRAM_power = ((SRAM * (E_BUF)) / RT) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
    # Offmem_power = ((Offmem * E_offmem) / RT) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
    # TOTALpower = MAC_power + SRAM_power + Offmem_power # mW
    # P_TPW = P_Throughput / TOTALpower
    # E_TPW = E_Throughput / TOTALpower

    # E_Throughput = np.round(E_Throughput,2)
    # MAC_power = np.round(MAC_power,2)
    # SRAM_power = np.round(SRAM_power,2)
    # Offmem_power = np.round(Offmem_power,2)
    # TOTALpower = np.round(TOTALpower,2)
    # P_TPW = np.round(P_TPW,2)
    # E_TPW = np.round(E_TPW,2)
    
    PP = Peak_P * Num_pod

    # rep_writer.writerow(["Batch", "Compute util", "P. Throughput (GFLOPS/S)", "E. Throughput (GFLOPS/S)",\
    #  "MAC Power (mW)", "SRAM Power (mW)", "Offmem Power (mW)", "Power Consumption (mW)", "P.Throughput/Watt (TOPS/W)", "E.Throughput/Watt (TOPS/W)"])
    # result_this_row = [Num_pod, PP, Comp_util, P_Throughput, E_Throughput, MAC_power, SRAM_power, Offmem_power, TOTALpower, P_TPW, E_TPW]
    result_this_row = [Num_pod, PP, RT, Total_E, EDP]
    rep_writer.writerow(result_this_row)


report_file.close()
overall_report_file.close()