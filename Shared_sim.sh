#!/bin/bash



# Newtile test
# topolist=("mobilenetv3_large")
# topolist=("mobilenetv3_large" "Googlenet" "DenseNet169" "resnet_fwd_mod_for16SA" "YOLOv3" "resnet152_for16SA" "BERT_base_10_MH" "BERT_large_64_MH" "BERT_large_256_MH")
topolist=("mobilenetv3_large" "DenseNet169" "resnet_fwd_mod_for16SA" "BERT_base_10_MH" "BERT_large_64_MH" "ViT_huge_16")
# topolist=("BERT_large_64_MH")
# SA_Pod_list=("32x32 4x4")
# SA_Pod_list=("32x32 8x8")
SA_Pod_list=("16x16 8x8" "32x32 4x4" "64x64 2x2")
# SA_Pod_list=("128x128 1x1")
# Podlist=("1x1" "2x2" "4x4" "8x8" "16x16" "32x32")
Batch_sz="1"

for topo in ${topolist[@]}
do
    for SA_Pod in "${SA_Pod_list[@]}"
    do  
        # echo $SA_Pod
        # nohup python3 Scale_out_sim_newtile.py ${topo} ${SA_Pod} "1" 1> /dev/null 2>&1 &
        # nohup python3 Shared_buffer_sim_newtile.py ${topo} ${SA_Pod} "1" 1> /dev/null 2>&1 &
        nohup python3 SAVector_sim_perf_mode.py ${topo} ${SA_Pod} ${Batch_sz} 1> /dev/null 2>&1 &
        # nohup python3 SOSA_rev_padding_offmem.py ${topo} ${SA_Pod} "1" 1> /dev/null 2>&1 &
        nohup python3 SOSA_rev_RT.py ${topo} ${SA_Pod} ${Batch_sz} 1> /dev/null 2>&1 &
        # nohup python3 TPUv4i.py ${topo} "128x128" "2x2" ${Batch_sz} 1> /dev/null 2>&1 &
        # nohup python3 Shared_buffer_sim_ideal.py ${topo} ${SA_Pod} ${Batch_sz} 1> /dev/null 2>&1 &
        # nohup python3 Scale_out_sim_faster.py ${topo} "128x128" "2x2" "1" 1> /dev/null 2>&1 &
        # nohup python3 Scale_up_sim_faster.py ${topo} "256x256" "1x1" ${Batch_sz} 1> /dev/null 2>&1 &
    done
done