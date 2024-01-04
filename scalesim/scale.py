import argparse
import os

from scalesim.scale_sim import scalesim
#scale_sim.py에서 scalesim클래스 임포트

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', metavar='Topology file', type=str,
                        default="./topologies/conv_nets/resnet_fwd_l1.csv",
                        help="Path to the topology file"
                        )
    #-t 인자 : topology 파일 경로 받음.
    parser.add_argument('-c', metavar='Config file', type=str,
                        default="./configs/google.cfg",
                        help="Path to the config file"
                        )
    #-c 인자 : config 파일 경로 받음.
    parser.add_argument('-p', metavar='log dir', type=str,
                        default=".\\test_runs",
                        #../test_runs
                        help="Path to log dir"
                        )
    #-p 인자 : log 파일 경로 받음.
    parser.add_argument('-i', metavar='input type', type=str,
                        default="gemm",
                        help="Type of input topology, gemm: MNK, conv: conv"
                        )
    #-i 입력 topology의 유형(컨볼루션, GEMM...)

    args = parser.parse_args()
    topology = args.t
    config = args.c
    logpath = args.p
    inp_type = args.i
    #변수에 입력받은 각 인자들 대입

    gemm_input = False
    if inp_type == 'gemm':
        gemm_input = True
    #-i옵션에 GEMM들어오면 True, 아니면 False로 세팅.
    
    s = scalesim(save_disk_space=False, verbose=True,
                config=config,topology=topology,
                input_type_gemm=gemm_input
                )
    #인자들을 이용해 scalesim클래스 초기화
    #save_disk_space=True면 trace생성.
    s.run_scale(top_path=logpath)
    #로그 경로를 입력하고 시뮬레이션 실행.
