#!/bin/bash


# topolist=("mobilenetv3_large" "Googlenet" "DenseNet169" "resnet_fwd_mod_for16SA" "YOLOv3" "resnet152_for16SA")

# for topo in ${topolist[@]}
# do
#     nohup python3 Moti_Scale_out_4mb.py ${topo} 1> /dev/null 2>&1 &
#     nohup python3 Moti_Scale_out_4mb_16.py ${topo} 1> /dev/null 2>&1 &
#     nohup python3 Moti_Scale_out_4mb_32.py ${topo} 1> /dev/null 2>&1 &
# done


# topolist=("BERT_base_10_MH" "BERT_large_64_MH" "BERT_large_256_MH")

# for topo in ${topolist[@]}
# do
#     nohup python3 Moti_Scale_out_mnk_4mb.py ${topo} 1> /dev/null 2>&1 &
#     nohup python3 Moti_Scale_out_mnk_4mb_16.py ${topo} 1> /dev/null 2>&1 &
#     nohup python3 Moti_Scale_out_mnk_4mb_32.py ${topo} 1> /dev/null 2>&1 &
# done

# Newtile test
# topolist=("mobilenetv3_large")
# topolist=("mobilenetv3_large" "Googlenet" "DenseNet169" "resnet_fwd_mod_for16SA" "YOLOv3" "resnet152_for16SA" "BERT_base_10_MH" "BERT_large_64_MH" "BERT_large_256_MH")
topolist=("mobilenetv3_large" "DenseNet169" "resnet_fwd_mod_for16SA" "BERT_base_10_MH" "BERT_large_64_MH" "ViT_huge_16")
# topolist=("YOLOv3")
SA_Pod_list=("128x128 1x1" "64x64 2x2" "32x32 4x4" "16x16 8x8" "8x8 16x16" "4x4 32x32")
# SA_Pod_list=("128x128 1x1")
# Podlist=("1x1" "2x2" "4x4" "8x8" "16x16" "32x32")

for topo in ${topolist[@]}
do
    for SA_Pod in "${SA_Pod_list[@]}"
    do  
        # echo $SA_Pod
        # nohup python3 Scale_out_sim_newtile.py ${topo} ${SA_Pod} "1" 1> /dev/null 2>&1 &
        # nohup python3 Shared_buffer_sim_ongoing.py ${topo} ${SA_Pod} "1" 1> /dev/null 2>&1 &
        nohup python3 Scale_out_sim_faster.py ${topo} ${SA_Pod} "1" 1> /dev/null 2>&1 &
        sleep 1
        # python3 Scale_out_sim_newtile.py ${topo} ${SA_Pod} 1
    done
done