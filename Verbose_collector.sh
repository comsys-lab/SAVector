#!/bin/bash

function Init_repfile {
    Repfile=$1
    
    [ -f $Repfile ] && rm $Repfile
    echo "Runtime,DRAM,SRAM,DRAM Reads,DRAM Writes,SRAM Reads,SRAM Writes,Comp Util,PE Util,Pod Util,SA_row,Active PE Ratio">> $Repfile
}

function sort_by_batch {
  filename=$1
  sort -t ',' -k13 -g $filename > sorted.csv
  mv sorted.csv $filename
}

function sort_by_numpod {
  filename=$1
  sort -t ',' -k12 -g $filename > sorted.csv
  mv sorted.csv $filename
}

function sort_by_SA {
  filename=$1
  # sort -t ',' -r -k11 -g $filename > sorted.csv
  # tail -n +2 $filename | sort -t ',' -r -k11 -g > sorted.csv
  { head -n 1 $filename && tail -n +2 $filename | sort -t ',' -r -k11 -g; } > sorted.csv
  mv sorted.csv $filename
}

function Gen_report {
    fname=$1 # ./Eval2_SAVector32/SAVector_Compact/YOLOv3/SAVector_32x32SA_4x4p_B32.txt
    Repfile=$2

    #echo "$f"
    #SA=${f##*/}
    A=$1
    # SA_size=${fname%%SA_*}

    # for SAVector_32x32SA_4x4p_B1.txt
    # SA_size="${fname##*SAVector_}"
    SA_size="${fname##*/}"  # "32x32SA_4x4p_B1.txt"
    SA_size="${SA_size%%SA_*}"      # "32x32"
    # echo $SA_size

    # SA_size=$(echo $SA_size | rev | cut -d '/' -f 1 | rev)
    # echo $SA_size # 128x128

    # Extract SA_row and SA_col
    SA_row=$(echo $SA_size | sed 's/\([0-9]*\)x\([0-9]*\)/\1/')
    SA_col=$(echo $SA_size | sed 's/\([0-9]*\)x\([0-9]*\)/\2/')
    # echo $SA_row # 128


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
    PEratio=`cat "$fname" | grep "PE ratio" | cut -c 19-`
    
    echo "$RT,$OverDRAM,$OverSRAM,$DRAMread,$DRAMwrite,$SRAMread,$SRAMwrite,$Computeutil,$PEutil,$PodUtil,$SA_row,$PEratio,$Batchsize">> $Repfile 
}


StartTime=$(date +%s)

# topologies=("mobilenetv3_large" "Googlenet" "DenseNet169" "resnet_fwd_mod_for16SA" "resnet152_for16SA" "YOLOv3" "BERT_base_10_MH" "BERT_large_64_MH" "BERT_large_256_MH") # DNN models to collect
# topologies=("mobilenetv3_large" "DenseNet169" "resnet_fwd_mod_for16SA" "resnet152_for16SA" "BERT_base_10_MH" "BERT_large_64_MH" "ViT_large_16" "ViT_huge_16" "YOLOv3") # DNN models to collect
# topologies=("mobilenetv3_large" "resnet_fwd_mod_for16SA" "resnet152_for16SA" "BERT_base_10_MH" "BERT_large_64_MH" "ViT_huge_16") # DNN models to collect
topologies=("mobilenetv3_large" "DenseNet169" "resnet_fwd_mod_for16SA" "BERT_base_10_MH" "BERT_large_64_MH" "ViT_huge_16") # DNN models to collect
# topologies=("mobilenetv3_large") # DNN models to collect

Report_dir="Moti_infbuf" # Moti_MACenergy, Eval1_rev, Eval2_rev
Output_dir="E_results"
mkdir -p ./${Output_dir}
Arch_name="Scaleout_oldtile" # Scaleout_oldtile, SAVector, SOSA_shared, Ideal, SOSA_RT, SOSA_pad, SAVector_perf, Scaleup, TPUv4i

# for d in ./Shared_results/Scaleout_oldtile_Compact/*
# for d in ./Moti_newtile/Scaleout_newtile_Compact/*

for d in ./${Report_dir}/${Arch_name}_Compact/*
do
    if [[ "${topologies[*]}" =~ $(basename "$d") ]]; then
        this_topo=${d##*/}

        Repname="Collect_${this_topo}"
        Repfile="./${Output_dir}/${Repname}.csv"
        # echo $Repfile
        
        Init_repfile $Repfile

        for f in $d/*
        do
            # echo $f
            # SA_this_file=${f%%SA*}
            # SA_this_file=$(echo $SA_this_file | rev | cut -d '/' -f 1 | rev) # e.g., 128x128

            # SA_this_file=$(echo $SA_this_file | sed 's|.*/\(.*\)|\1|')
            # echo $SA_this_file
            # echo $f
            # if [ "$Arch_name" == "SAVector" ]; then
            #   SA_this_file="${f##*/}"  # "32x32SA_4x4p_B1.txt"
            #   SA_this_file="${f##*SAVector_}"
            #   SA_this_file="${SA_this_file%%SA_*}"      # "32x32"
            #   if [ "$SA_this_file" == "32x32" ]; then
            #           Gen_report $f $Repfile
            #   fi
            # else
            #   Gen_report $f $Repfile
            # fi

            Gen_report $f $Repfile
        done

        sort_by_SA $Repfile
        # sort_by_batch $Repfile

        # python3 Energy_calculator.py $this_topo $Repname $Repfile $Output_dir
        # python3 TPW_calculator.py $this_topo $Repname $Repfile $Output_dir
        # python3 TPW_calculator_eval3.py $this_topo $Repname $Repfile $Output_dir
        python3 Verbose_calculator.py $this_topo $Repname $Repfile $Output_dir

    else
        continue
    fi
done

# python3 Energy_gather.py $Output_dir
python3 Verbose_gather.py $Output_dir


EndTime=$(date +%s)
echo "It takes $(($EndTime - $StartTime)) seconds to complete this task."