import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

E_TPW=[]
csvs = ["BERT_base_10","BERT_base_50","BERT_base_100","BERT_base_200","BERT_base_300"]
#csvs = ["mobilenetv3_small","mobilenetv3_large","resnet_fwd_mod","Googlenet","YOLOv3"]
#csvs = ["BERT_base_10","BERT_base_50","BERT_base_100","BERT_base_200","BERT_base_300","mobilenetv3_small","mobilenetv3_large","resnet_fwd_mod","Googlenet","YOLOv3"]

for topo in csvs:
    csv_name="./TPWresults/"+topo+"_result.csv"
    df = pd.read_csv(csv_name, header=0)
    df2 = df[['SA_row', 'SA_col','E.Throughput/Watt (TOPS/W)']]
    df2 = df2.sort_values(by=['SA_row','SA_col'])
    E_TPW_this = df2['E.Throughput/Watt (TOPS/W)'].to_numpy()
    E_TPW.append(E_TPW_this)

print(E_TPW)
# for item in E_TPW:
#     print(item[0])
# 0/0

temp=E_TPW[0]
for i in range(len(E_TPW)-1):
    temp = temp+E_TPW[i+1]
temp=np.divide(temp,len(csvs)).reshape(len(temp),1)
print(temp.shape)

SAs=(df2[['SA_row', 'SA_col']].to_numpy()).astype('str')
out=np.concatenate((SAs,temp),axis=1)
print(out)

out_df=pd.DataFrame(out)
out_df.columns = ['SA_row', 'SA_col','E.Throughput/Watt (TOPS/W)']
out_df.to_csv('./TPWresults/AVG_BERTs_result.csv',index=False)