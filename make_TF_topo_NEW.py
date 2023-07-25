import csv
import os

# BERT 다시 만듬: M=if_row, N=fil_col, K=if_col & fil_row였음.

toponame=input('topo name (file name without \'.csv\'): ')
topotype=input('topo type (TF, BERT, GPT): ')
vocab=int(input('vocabulary size: '))
embed=int(input('embedding size (d_model): '))
head=int(input('number of heads: '))
assert embed % head == 0
dk=int(embed/head)

fname='/home/choi/scalesimv2/topologies/TFs_mnk/'+toponame+'.csv'
f=open(fname,'w',newline='')
wr=csv.writer(f)
if topotype=='TF':
    #encoder
    wr.writerow(['Layer','M','N','K',''])
    wr.writerow(['MH_get_QKV',vocab,embed,dk,''])
    wr.writerow(['MH_QxKT',vocab,dk,vocab,''])
    wr.writerow(['MH_QxKTxV',vocab,vocab,dk,''])
    wr.writerow(['MH_Concat_Linear',vocab,embed,embed,''])
    wr.writerow(['FF1',vocab,embed,4*embed,''])
    wr.writerow(['FF2',vocab,4*embed,embed,''])
    #decoder
    wr.writerow(['MH_get_QKV',vocab,embed,dk,''])
    wr.writerow(['MH_QxKT',vocab,dk,vocab,''])
    wr.writerow(['MH_QxKTxV',vocab,vocab,dk,''])
    wr.writerow(['MH_Concat_Linear',vocab,embed,embed,''])
    wr.writerow(['MH_get_QKV',vocab,embed,dk,''])
    wr.writerow(['MH_QxKT',vocab,dk,vocab,''])
    wr.writerow(['MH_QxKTxV',vocab,vocab,dk,''])
    wr.writerow(['MH_Concat_Linear',vocab,embed,embed,''])
    wr.writerow(['FF1',vocab,embed,4*embed,''])
    wr.writerow(['FF2',vocab,4*embed,embed,''])
elif topotype=='BERT':
    #encoder
    wr.writerow(['Layer','M','N','K',''])
    for i in range(head):
        wr.writerow(['MH_get_Q_head{}'.format(i),vocab,dk,embed,''])
        wr.writerow(['MH_get_K_head{}'.format(i),vocab,dk,embed,''])
        wr.writerow(['MH_get_V_head{}'.format(i),vocab,dk,embed,''])
        wr.writerow(['MH_QxKT_head{}'.format(i),vocab,vocab,dk,''])
        wr.writerow(['MH_QxKTxV_head{}'.format(i),vocab,dk,vocab,''])
    wr.writerow(['MH_Concat_Linear',vocab,embed,embed,''])
    wr.writerow(['FF1',vocab,4*embed,embed,''])
    wr.writerow(['FF2',vocab,embed,4*embed,''])
elif topotype=='GPT':
    #decoder
    wr.writerow(['Layer','M','N','K',''])
    for i in range(head):
        wr.writerow(['Masked_MH_get_Q_head{}'.format(i),vocab,dk,embed,''])
        wr.writerow(['Masked_MH_get_K_head{}'.format(i),vocab,dk,embed,''])
        wr.writerow(['Masked_MH_get_V_head{}'.format(i),vocab,dk,embed,''])
        wr.writerow(['Masked_MH_QxKT_head{}'.format(i),vocab,vocab,dk,''])
        wr.writerow(['Masked_MH_QxKTxV_head{}'.format(i),vocab,dk,vocab,''])
    wr.writerow(['Masked_MH_Concat_Linear',vocab,embed,embed,''])
    # for i in range(head):
    #     wr.writerow(['MH_get_Q_head{}'.format(i),vocab,dk,embed,''])
    #     wr.writerow(['MH_get_K_head{}'.format(i),vocab,dk,embed,''])
    #     wr.writerow(['MH_get_V_head{}'.format(i),vocab,dk,embed,''])
    #     wr.writerow(['MH_QxKT_head{}'.format(i),vocab,vocab,dk,''])
    #     wr.writerow(['MH_QxKTxV_head{}'.format(i),vocab,dk,vocab,''])
    # wr.writerow(['MH_Concat_Linear',vocab,embed,embed,''])
    wr.writerow(['FF1',vocab,embed,4*embed,''])
    wr.writerow(['FF2',vocab,4*embed,embed,''])
f.close()