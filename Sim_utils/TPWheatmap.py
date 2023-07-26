import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('topo', help='Set topology file')
args = parser.parse_args()
toponame= args.topo #topology 이름 (e.g.,alexnet)

csv_name = "./TPWresults/" + toponame + "_result.csv"
df = pd.read_csv(csv_name, header=0)
df2 = df[['SA_row', 'SA_col','E.Throughput/Watt (TOPS/W)']]
df2 = df2.sort_values(by=['SA_row','SA_col'])
df2 = df2[(df2['SA_row']>=8)&(df2['SA_col']>=8)]
#df2.head()

pivot_df = df2.pivot('SA_row','SA_col','E.Throughput/Watt (TOPS/W)')
#print(pivot_df.head())

# heatmap by plt.pcolor()

plt.rcParams['figure.figsize'] = [10, 8]
plt.pcolor(pivot_df)
plt.xticks(np.arange(0.5, len(pivot_df.columns), 1), pivot_df.columns)
plt.yticks(np.arange(0.5, len(pivot_df.index), 1), pivot_df.index)
#plt.ylim([8,128])
plt.title('Heatmap by plt.pcolor()', fontsize=20)
plt.xlabel('SA_col', fontsize=14)
plt.ylabel('SA_row', fontsize=14)
plt.coolwarm()
plt.colorbar()
figname=str(toponame+'_heat')
plt.savefig(figname)