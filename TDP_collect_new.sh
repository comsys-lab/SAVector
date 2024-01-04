#!/bin/bash

function Init_repfile {
    Repfile=$1
    
    [ -f $Repfile ] && rm $Repfile
    echo "Runtime,DRAM,SRAM,DRAM Reads,DRAM Writes,SRAM Reads,SRAM Writes,Comp Util,PE Util,Pod Util,Batch,num Pod">> $Repfile
}

function sort_by_batch {
  filename=$1
  sort -t ',' -k11 -g $filename > sorted.csv
  mv sorted.csv $filename
}

function sort_by_numpod {
  filename=$1
  sort -t ',' -k12 -g $filename > sorted.csv
  mv sorted.csv $filename
}

function Gen_report {
    fname=$1
    Repfile=$2

    #echo "$f"
    #SA=${f##*/}
    A=$1
    SA_size=$(echo $A | sed 's/SAVector_\([0-9]*\)x\([0-9]*\)SA.*/\1x\2/')
    SA_size=$(echo $SA_size | sed 's|.*/\(.*\)|\1|')

    # Extract SA_row and SA_col
    SA_row=$(echo $SA_size | sed 's/\([0-9]*\)x\([0-9]*\)/\1/')
    SA_col=$(echo $SA_size | sed 's/\([0-9]*\)x\([0-9]*\)/\2/')

    # Extract pod_dim
    pod_dim=$(echo $A | sed 's/.*_\([0-9]*\)x\([0-9]*\)p_.*/\1x\2/')

    # Extract pod_row and pod_col
    pod_row=$(echo $pod_dim | sed 's/\([0-9]*\)x\([0-9]*\)/\1/')
    pod_col=$(echo $pod_dim | sed 's/\([0-9]*\)x\([0-9]*\)/\2/')

    # Calculate num_pod
    num_pod=$((pod_row * pod_col))

    RT=`cat "$fname" | grep runtime | cut -c 24-`
    OverDRAM=`cat "$fname" | grep "all DRAM" | cut -c 21-`
    OverSRAM=`cat "$fname" | grep "all SRAM" | cut -c 21-`
    #PEutil=`cat "$fname" | grep "PE Util" | cut -c 24-`
    PEutil=`cat "$fname" | grep "Weighted Util" | cut -c 15-`
    Computeutil=`cat "$fname" | grep "Compute" | cut -c 14-`
    PodUtil=`cat "$fname" | grep "Weighted Pod Util" | cut -c 19-`
    DRAMread=`cat "$fname" | grep "DRAM reads" | cut -c 12-`
    DRAMwrite=`cat "$fname" | grep "DRAM writes" | cut -c 13-`
    SRAMread=`cat "$fname" | grep "SRAM reads" | cut -c 12-`
    SRAMwrite=`cat "$fname" | grep "SRAM writes" | cut -c 13-`
    Batchsize=`cat "$fname" | grep "Batch" | cut -c 12-`
    
    echo "$RT,$OverDRAM,$OverSRAM,$DRAMread,$DRAMwrite,$SRAMread,$SRAMwrite,$Computeutil,$PEutil,$PodUtil,$Batchsize,$num_pod,$pod_row,$pod_col">> $Repfile 
}


StartTime=$(date +%s)

# topologies=("resnet_fwd_mod_for16SA" "BERT_large_64_MH") # DNN models to collect
# topologies=("mobilenetv3_large" "Googlenet" "DenseNet169" "resnet_fwd_mod_for16SA" "resnet152_for16SA" "YOLOv3" "BERT_base_10_MH" "BERT_large_64_MH" "BERT_large_256_MH") # DNN models to collect
topologies=("BERT_TDP_4096") # DNN models to collect
SA_list=("16x16" "32x32" "64x64")

for this_SA in ${SA_list[@]}
do
    for d in ./TDP_rev_SAVector_WDB/SAVector_Compact/*
    do
        if [[ "${topologies[*]}" =~ $(basename "$d") ]]; then
            this_topo=${d##*/}

            Repname="SAVector_collect_${this_topo}_${this_SA}SA"
            Repfile="./TDP_calculated_results/${Repname}.csv"
            
            Init_repfile $Repfile

            for f in $d/*
            do
                SA_this_file=$(echo $f | sed 's/SAVector_\([0-9]*\)x\([0-9]*\)SA.*/\1x\2/')
                SA_this_file=$(echo $SA_this_file | sed 's|.*/\(.*\)|\1|')
                # echo $SA_this_file

                # echo $SA_this_file
                if [ "$SA_this_file" == "$this_SA" ]; then
                    Gen_report $f $Repfile
                fi
            done

            sort_by_numpod $Repfile

            python3 TPW_calculator_TDP_EDP.py $this_topo $Repname $Repfile $this_SA

        else
            continue
        fi
    done
done


# TPW_folder="./TPWresults"
# python3 TPW_gen_excel.py $TPW_folder

EndTime=$(date +%s)
echo "It takes $(($EndTime - $StartTime)) seconds to complete this task."