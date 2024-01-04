import numpy as np
import csv
import argparse

import os

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

parser = argparse.ArgumentParser(description='')
parser.add_argument('topo', help='Set topology file')
parser.add_argument('repname', help='Set report file to read')
parser.add_argument('repfile', help='Set report file to read')
parser.add_argument('outdir', help='Set report file to read')
# parser.add_argument('SAsize', help='Set report file to read')
args = parser.parse_args()
toponame= args.topo #topology name (e.g.,alexnet)

if "BERT" in toponame:
    mnk_flag = True
else:
    mnk_flag = False

MACs_row = 256 # number of total MAC units
MACs_col = 256

# SA_dim = args.SAsize
# SA_row, SA_col = map(int, SA_dim.split('x'))

E_MAC = 0.56 # pJ
E_MAC_static = 0.017 # pJ
E_BUF = 3.16 # pJ
E_BUF_large = 8.06 # pJ
E_tinyBUF = 0.15 # pJ
E_offmem = 31.2 # pJ, HBM=31.2, DDR4=140, GDDR6=112
Clock_freq = 1000000000 # 1GHz

# Report file to open
repname = args.repfile
report_file = open(repname,'r')
rep_reader = csv.reader(report_file)
rep_header = next(rep_reader)

# Report file to write
repname = args.repname
outdir = args.outdir
createDirectory('./'+outdir)
over_repname = './'+outdir + '/' + repname + '.csv'
overall_report_file = open(over_repname, 'w', newline='')
rep_writer = csv.writer(overall_report_file)

#Batch,Cutil,PeakT,EffT,MACP,SRAMP,OffP,P,EffTPW
# rep_writer.writerow(["SA dim", "Energy (mJ)", "MAC_E (mJ)", "SRAM_E (mJ)","Offmem_E (mJ)"])
rep_writer.writerow(["SA dim", "Compute util", "P. Throughput (GFLOPS/S)", "E. Throughput (GFLOPS/S)",\
 "MAC Power (mW)", "SRAM Power (mW)", "Offmem Power (mW)", "Power Consumption (mW)", "P.Throughput/Watt (TOPS/W)", "E.Throughput/Watt (TOPS/W)", "Energy"])

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
    # Batch=int(rows[10])
    SA_row=int(rows[10])
    
    # P_Throughput = (2 * MACs_row * MACs_col) # GFLOPS
    # E_Throughput = P_Throughput * Comp_util * 0.01 # GFLOPS
    # MAC_power = ((E_MAC * MACs_row * MACs_col) + (E_MAC_static * MACs_row * MACs_col)) * (Pod_util * 0.01) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
    # if 'Scale_up' in repname:
    #     SRAM_power = ((SRAM * (E_BUF_large)) / RT) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
    # else:
    #     SRAM_power = ((SRAM * (E_BUF + E_tinyBUF)) / RT) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
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
    
    
    # MAC_E = MAC_power * (RT / np.power(10,3))
    # SRAM_E = SRAM_power * (RT / np.power(10,3))
    # Offmem_E = Offmem_power * (RT / np.power(10,3))
    # Total_E = MAC_E + SRAM_E + Offmem_E
    
    # if SA_row == 128:
    #     E_BUF = 8.85
    # elif SA_row == 64:
    #     E_BUF = 4.19
    # elif SA_row == 32:
    #     E_BUF = 1.84
    # elif SA_row == 16:
    #     E_BUF = 0.7
    # elif SA_row == 8:
    #     E_BUF = 0.34
    # elif SA_row == 4:
    #     E_BUF = 0.23
    # else:
    #     print('error SA dim')
        
    if SA_row >= 128:
        E_BUF = 8.85
    elif SA_row == 64:
        E_BUF = 4.69
    elif SA_row == 32:
        E_BUF = 3.16
    elif SA_row == 16:
        E_BUF = 3.16
    elif SA_row == 8:
        E_BUF = 3.16
    elif SA_row == 4:
        E_BUF = 3.16
    else:
        print('error SA dim')
        
    
    MAC_E = (E_MAC * MACs_row * MACs_col) * (Pod_util * 0.01) * (RT * (Comp_util * 0.01)) + ((E_MAC_static * MACs_row * MACs_col) * (Pod_util * 0.01) * RT) # cycle마다의 MAC power * 실제 연산 수행한 cycles
    MAC_E = np.round(MAC_E/np.power(10,9),2)
    SRAM_E = (E_BUF+E_tinyBUF) * SRAM
    SRAM_E = np.round(SRAM_E/np.power(10,9),2)
    Offmem_E = Offmem * E_offmem
    Offmem_E = np.round(Offmem_E/np.power(10,9),2)
    # Offmem_E = 0
    
    Total_E = np.round(MAC_E + SRAM_E + Offmem_E,2)
    
    
    # Modify from here #
    P_Throughput = (2 * MACs_row * MACs_col) # GFLOPS
    E_Throughput = P_Throughput * (Comp_util * 0.01) # GFLOPS
    MAC_power = ((E_MAC * MACs_row * MACs_col * (Comp_util * 0.01)) + (E_MAC_static * MACs_row * MACs_col)) * (Pod_util * 0.01) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
    if SA_row >= 128:
        SRAM_power = ((SRAM * (E_BUF)) / RT) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
    else:
        SRAM_power = ((SRAM * (E_BUF + E_tinyBUF)) / RT) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
    Offmem_power = ((Offmem * E_offmem) / RT) / np.power(10,12) * Clock_freq * np.power(10,3)# mW
    TOTALpower = MAC_power + SRAM_power + Offmem_power # mW
    P_TPW = P_Throughput / TOTALpower
    E_TPW = E_Throughput / TOTALpower

    E_Throughput = np.round(E_Throughput,2)
    MAC_power = np.round(MAC_power,2)
    SRAM_power = np.round(SRAM_power,2)
    Offmem_power = np.round(Offmem_power,2)
    TOTALpower = np.round(TOTALpower,2)
    P_TPW = np.round(P_TPW,2)
    E_TPW = np.round(E_TPW,2)
    
    # PP = Peak_P * Num_pod
    
    
    
    if "BERT_base" in toponame:
        MAC_E = np.round(MAC_E * 12,2)
        SRAM_E = np.round(SRAM_E * 12,2)
        Offmem_E = np.round(Offmem_E * 12,2)
        Total_E = np.round(Total_E * 12,2)
    elif "BERT_large" in toponame:
        MAC_E = np.round(MAC_E * 24,2)
        SRAM_E = np.round(SRAM_E * 24,2)
        Offmem_E = np.round(Offmem_E * 24,2)
        Total_E = np.round(Total_E * 24,2)

    # rep_writer.writerow(["Batch", "Compute util", "P. Throughput (GFLOPS/S)", "E. Throughput (GFLOPS/S)",\
    #  "MAC Power (mW)", "SRAM Power (mW)", "Offmem Power (mW)", "Power Consumption (mW)", "P.Throughput/Watt (TOPS/W)", "E.Throughput/Watt (TOPS/W)"])
    # result_this_row = [SA_row, Total_E, MAC_E, SRAM_E, Offmem_E]
    result_this_row = [SA_row, Comp_util, P_Throughput, E_Throughput, MAC_power, SRAM_power, Offmem_power, TOTALpower, P_TPW, E_TPW, Total_E]

    rep_writer.writerow(result_this_row)


report_file.close()
overall_report_file.close()