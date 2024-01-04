#!/bin/bash


batch=1

# topolist=("resnet152_for16SA")
# topolist=("BERT_large_256_MH")
# topolist=("DenseNet169" "resnet_fwd_mod_for16SA" "BERT_large_64_MH")
# topolist=("resnet152_for16SA" "YOLOv3" "BERT_large_256_MH")
topolist=("BERT_TDP_4096")

#"1x1" "1x2" "2x1" "2x2" "2x4" "4x2" "4x4" "4x8" "8x4" "8x8" "8x16" "16x8" "16x16" "16x32" "32x16" "32x32"

# for 16x16 SA
podlist=("8x8" "8x16" "16x8" "16x16" "16x32" "32x16" "32x32")

for topo in ${topolist[@]}
do
    for poddim in ${podlist[@]}
    do
        # nohup python3 SAVector_TDP.py ${topo} 16x16 ${poddim} ${batch} 1> /dev/null 2>&1 &
        nohup python3 SAVector_sim_faster.py ${topo} 16x16 ${poddim} ${batch} 1> /dev/null 2>&1 &
    done
done

# for 32x32 SA
podlist=("4x4" "4x8" "8x4" "8x8" "8x16" "16x8" "16x16" "16x32" "32x16")

for topo in ${topolist[@]}
do
    for poddim in ${podlist[@]}
    do
        # nohup python3 SAVector_TDP.py ${topo} 32x32 ${poddim} ${batch} 1> /dev/null 2>&1 &
        nohup python3 SAVector_sim_faster.py ${topo} 32x32 ${poddim} ${batch} 1> /dev/null 2>&1 &
    done
done

# for 64x64 SA
podlist=("2x2" "2x4" "4x2" "4x4" "4x8" "8x4" "8x8" "8x16" "16x8" "16x16")

for topo in ${topolist[@]}
do
    for poddim in ${podlist[@]}
    do
        # nohup python3 SAVector_TDP.py ${topo} 64x64 ${poddim} ${batch} 1> /dev/null 2>&1 &
        nohup python3 SAVector_sim_faster.py ${topo} 64x64 ${poddim} ${batch} 1> /dev/null 2>&1 &
    done
done