import os

from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo
from scalesim.single_layer_sim import single_layer_sim as layer_sim
#scale_config.py에서 scale_config클래스를 cfg이름으로 임포트
#topology_utils.py에서 topologies클래스를 topo이름으로 임포트
#single_layer_sim.py에서 single_layer_sim클래스를 layer_sim이름으로 임포트

class simulator:
    def __init__(self):
        self.conf = cfg()
        self.topo = topo()

        self.top_path = "./"
        self.verbose = True
        self.save_trace = True

        self.num_layers = 0

        self.single_layer_sim_object_list = []

        self.params_set_flag = False
        self.all_layer_run_done = False
    # 초기화

    #
    def set_params(self,
                   config_obj=cfg(),
                   topo_obj=topo(),
                   top_path="./",
                   verbosity=True,
                   save_trace=True
                   ):

        self.conf = config_obj
        self.topo = topo_obj

        self.top_path = top_path
        self.verbose = verbosity
        self.save_trace = save_trace

        # Calculate inferrable parameters here
        self.num_layers = self.topo.get_num_layers()

        self.params_set_flag = True
    # scale_sim.py에서 run_scale멤버함수로 위 set_params메소드 실행.
    # 여러 파라미터들 설정.


    #
    def run(self):
        assert self.params_set_flag, 'Simulator parameters are not set'
        # params_set_flag==False면 중지.

        # 1. Create the layer runners for each layer
        for i in range(self.num_layers):
            this_layer_sim = layer_sim() #single_layer_sim(=layer_sim) 객체 만듬
            this_layer_sim.set_params(layer_id=i,
                                 config_obj=self.conf,
                                 topology_obj=self.topo,
                                 verbose=self.verbose) #single_layer_sim에서 set_params메소드 통해서 설정

            self.single_layer_sim_object_list.append(this_layer_sim) #만든 레이어 객체를 리스트에 추가.

        if not os.path.isdir(self.top_path):
            cmd = 'mkdir ' + self.top_path
            os.system(cmd) #top_path 폴더 없으면 만듬.

        report_path = self.top_path + '/' + self.conf.get_run_name() #report경로 설정.

        if not os.path.isdir(report_path):
            cmd = 'mkdir ' + report_path
            os.system(cmd) #report경로 없으면 만듬.

        self.top_path = report_path #top_path를 report_path로 업데이트

        # 2. Run each layer
        # TODO: This is parallelizable
        for single_layer_obj in self.single_layer_sim_object_list: #리스트 내 layer들에 대해..

            if self.verbose:
                layer_id = single_layer_obj.get_layer_id()
                print('\nRunning Layer ' + str(layer_id))
            #verbose=True면, 실행 중인 레이어 이름 표시.
            single_layer_obj.run()
            #single_layer_obj객체의 run() 실행.

            if self.verbose:
                comp_items = single_layer_obj.get_compute_report_items()
                comp_cycles = comp_items[0]
                stall_cycles = comp_items[1]
                util = comp_items[2]
                mapping_eff = comp_items[3]
                print('Compute cycles: ' + str(comp_cycles))
                print('Stall cycles: ' + str(stall_cycles))
                print('Overall utilization: ' + "{:.2f}".format(util) +'%')
                print('Mapping efficiency: ' + "{:.2f}".format(mapping_eff) +'%')

                avg_bw_items = single_layer_obj.get_bandwidth_report_items()
                avg_ifmap_bw = avg_bw_items[3]
                avg_filter_bw = avg_bw_items[4]
                avg_ofmap_bw = avg_bw_items[5]
                print('Average IFMAP DRAM BW: ' + "{:.3f}".format(avg_ifmap_bw) + ' words/cycle')
                print('Average Filter DRAM BW: ' + "{:.3f}".format(avg_filter_bw) + ' words/cycle')
                print('Average OFMAP DRAM BW: ' + "{:.3f}".format(avg_ofmap_bw) + ' words/cycle')
            #verbose=True면, 위 정보들 표시.

            if self.save_trace:
                if self.verbose:
                    print('Saving traces: ', end='')
                single_layer_obj.save_traces(self.top_path) #save_trace=True면 save_traces메소드 실행(top_path경로에)
                if self.verbose:
                    print('Done!') #verbose=True면 추가 설명 출력.

        self.all_layer_run_done = True #전부 실행하면 done플래그 True

        self.generate_reports() #아래 generate_reports() 실행.

    #
    def generate_reports(self):
        assert self.all_layer_run_done, 'Layer runs are not done yet'
        # 위에서 layer실행 전부 완료 완됐으면 중지.

        compute_report_name = self.top_path + '/COMPUTE_REPORT.csv'
        compute_report = open(compute_report_name, 'w')
        header = 'LayerID, Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %,\n'
        compute_report.write(header)
        # compute 리포트 파일명 설정, 해당 파일 쓰기로 열고, 헤더 스트링 만들고 쓰기.

        bandwidth_report_name = self.top_path + '/BANDWIDTH_REPORT.csv'
        bandwidth_report = open(bandwidth_report_name, 'w')
        header = 'LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg OFMAP SRAM BW, '
        header += 'Avg IFMAP DRAM BW, Avg FILTER DRAM BW, Avg OFMAP DRAM BW,\n'
        bandwidth_report.write(header)
        # bandwidth 리포트 파일명 설정, 쓰기로 열고, 헤더 2가지 스트링 만들고 쓰기.

        detail_report_name = self.top_path + '/DETAILED_ACCESS_REPORT.csv'
        detail_report = open(detail_report_name, 'w')
        header = 'LayerID, '
        header += 'SRAM IFMAP Start Cycle, SRAM IFMAP Stop Cycle, SRAM IFMAP Reads, '
        header += 'SRAM Filter Start Cycle, SRAM Filter Stop Cycle, SRAM Filter Reads, '
        header += 'SRAM OFMAP Start Cycle, SRAM OFMAP Stop Cycle, SRAM OFMAP Writes, '
        header += 'DRAM IFMAP Start Cycle, DRAM IFMAP Stop Cycle, DRAM IFMAP Reads, '
        header += 'DRAM Filter Start Cycle, DRAM Filter Stop Cycle, DRAM Filter Reads, '
        header += 'DRAM OFMAP Start Cycle, DRAM OFMAP Stop Cycle, DRAM OFMAP Writes,\n'
        detail_report.write(header)
        # detail 리포트도 마찬가지.

        for lid in range(len(self.single_layer_sim_object_list)):
            single_layer_obj = self.single_layer_sim_object_list[lid]
            compute_report_items_this_layer = single_layer_obj.get_compute_report_items()
            log = str(lid) +', '
            log += ', '.join([str(x) for x in compute_report_items_this_layer])
            log += ',\n'
            compute_report.write(log)

            bandwidth_report_items_this_layer = single_layer_obj.get_bandwidth_report_items()
            log = str(lid) + ', '
            log += ', '.join([str(x) for x in bandwidth_report_items_this_layer])
            log += ',\n'
            bandwidth_report.write(log)

            detail_report_items_this_layer = single_layer_obj.get_detail_report_items()
            log = str(lid) + ', '
            log += ', '.join([str(x) for x in detail_report_items_this_layer])
            log += ',\n'
            detail_report.write(log)

        compute_report.close()
        bandwidth_report.close()
        detail_report.close()

    #
    def get_total_cycles(self):
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        total_cycles = 0
        for layer_obj in self.single_layer_sim_object_list:
            cycles_this_layer = int(layer_obj.get_compute_report_items[0])
            total_cycles += cycles_this_layer

        return total_cycles



