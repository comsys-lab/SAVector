#!/bin/bash

toponame=$1

StartTime=$(date +%s)

[ -f ./TPWresults/$toponame.csv ] && rm ./TPWresults/$toponame.csv
echo "SA_row, SA_col, RT, OverDRAM, OverSRAM, PEutil, Computeutil, DRAMread, DRAMwrite, SRAMread, SRAMwrite">> ./TPWresults/$toponame.csv

for d in ./SAreports_Compact/*
do
    this_topo=${d##*/}
    #echo "$this_topo"
    if [ $this_topo == $toponame ];then
        for f in $d/*
        do
            #echo "$f"
            #SA=${f##*/}
            SA=`basename -s ".txt" "$f"`
            SA_row=`echo $SA | cut -d 'x' -f1`
            SA_col=`echo $SA | cut -d 'x' -f2`
            RT=`cat "$f" | grep runtime | cut -c 24-`
            #OverDRAM=`cat "$f" | grep "all DRAM" | cut -c 21-`
            OverSRAM=`cat "$f" | grep "all SRAM" | cut -c 21-`
            #PEutil=`cat "$f" | grep "PE Util" | cut -c 24-`
            #Computeutil=`cat "$f" | grep "Compute" | cut -c 14-`
            #DRAMread=`cat "$f" | grep "DRAM reads" | cut -c 12-`
            #DRAMwrite=`cat "$f" | grep "DRAM writes" | cut -c 13-`
            #SRAMread=`cat "$f" | grep "SRAM reads" | cut -c 12-`
            #SRAMwrite=`cat "$f" | grep "SRAM writes" | cut -c 13-`
            #echo "$SA_row,$SA_col,$RT,$OverDRAM,$OverSRAM,$PEutil,$Computeutil,$DRAMread,$DRAMwrite,$SRAMread,$SRAMwrite">> ./TPWresults/$toponame.csv
            echo "$SA_row,$SA_col,$RT,$OverSRAM">> ./TPWresults/$toponame.csv
        done
    fi
done

python3 Wutil.py $toponame

python3 TPWheatmap.py $toponame

EndTime=$(date +%s)
echo "It takes $(($EndTime - $StartTime)) seconds to complete this task."