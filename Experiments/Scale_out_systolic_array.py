import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
from datetime import datetime

from scalesim.compute.operand_matrix import operand_matrix as opmat
from scalesim.topology_utils import topologies
from scalesim.scale_config import scale_config

from scalesim.compute.systolic_compute_os import systolic_compute_os
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.compute.systolic_compute_is import systolic_compute_is
from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad as mem_dbsp

import os

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def mkcfg(fname, dataflow, Arr_H, Arr_W, IF_SRAM, Fil_SRAM, OF_SRAM):
    fwrite=open(fname,'w')

    #make cfg file
    fwrite.write('\
    [general]\n\
    run_name = temporal\n\
    \n\
    [architecture_presets]\n\
    ArrayHeight:    {}\n\
    ArrayWidth:     {}\n\
    IfmapSramSzkB:    {}\n\
    FilterSramSzkB:   {}\n\
    OfmapSramSzkB:    {}\n\
    IfmapOffset:    0\n\
    FilterOffset:   10000000\n\
    OfmapOffset:    20000000\n\
    Dataflow : {}\n\
    Bandwidth : 10\n\
    MemoryBanks: 1\n\
    \n\
    [run_presets]\n\
    InterfaceBandwidth: CALC'.format(Arr_H, Arr_W, IF_SRAM, Fil_SRAM, OF_SRAM, dataflow))
    fwrite.close()
    return


class scaled_out_simulator:
    def __init__(self):
        self.topo_obj = topologies()
        self.single_arr_cfg = scale_config()

        self.grid_rows = 1
        self.grid_cols = 1
        self.dataflow = 'os'

        # Stats objects
        self.stats_compute_cycles = np.ones(1) * -1
        self.stats_ifmap_dram_reads = np.ones(1) * -1
        self.stats_ifmap_dram_start_cycl = np.ones(1) * -1
        self.stats_ifmap_dram_end_cycl = np.ones(1) * -1

        self.stats_filter_dram_reads = np.ones(1) * -1
        self.stats_filter_dram_start_cycl = np.ones(1) * -1
        self.stats_filter_dram_end_cycl = np.ones(1) * -1

        self.stats_ofmap_dram_reads = np.ones(1) * -1
        self.stats_ofmap_dram_start_cycl = np.ones(1) * -1
        self.stats_ofmap_dram_end_cycl = np.ones(1) * -1

        self.overall_compute_cycles_per_layers = []
        self.overall_util_perc_per_layer = []

        self.overall_compute_cycles_all_layers = 0
        self.overall_util_perc_all_layer = 0

        self.total_ifmap_dram_reads = []
        self.total_filter_dram_reads = []
        self.total_ofmap_dram_writes = []

        ########################### SRAM 
        self.stats_ifmap_sram_reads = np.ones(1) * -1
        self.stats_filter_sram_reads = np.ones(1) * -1
        self.stats_ofmap_sram_writes = np.ones(1) * -1

        self.total_ifmap_sram_reads = []
        self.total_filter_sram_reads = []
        self.total_ofmap_sram_writes = []
        ##############################################

        ########## For new tiling strategy ##########
        self.ifmap_tiles_each_pod_acc = []
        self.filter_tiles_each_pod_acc = []
        ########## For new utilization (mapping eff & pod eff) ########
        self.OVER_util = []
        self.POD_util = []
        #############################################

        # Flags
        self.params_valid = False
        self.all_grids_done = False
        self.metrics_ready = False

    #
    def set_params(self,
                    topology_filename='./files/tutorial3_topofile.csv',
                    single_arr_config_file='./files/single_arr_config.cfg',
                    grid_rows=1, grid_cols=1,
                    dataflow = 'os',
                    mnk_input= False,
                    num_pods=1
                    ):

        # 23.07.17 Set number of pods. #####
        self.num_pods = num_pods
        ####################################

        # Blank 1. Read the input files 
        self.topo_obj = topologies() 
        self.topo_obj.load_arrays(topology_filename, mnk_inputs=mnk_input)
        num_layers=self.topo_obj.get_num_layers() 

        self.single_arr_cfg = scale_config()
        self.single_arr_cfg.read_conf_file(single_arr_config_file) 
        # <insert code here>

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols 

        num_arrays = grid_rows * grid_cols 
        self.stats_compute_cycles = np.ones((num_layers, num_arrays)) * -1 
        #

        self.stats_ifmap_dram_reads = np.ones((num_layers, num_arrays)) * -1
        self.stats_ifmap_dram_start_cycl = np.ones((num_layers, num_arrays)) * -1
        self.stats_ifmap_dram_end_cycl = np.ones((num_layers, num_arrays)) * -1

        self.stats_filter_dram_reads = np.ones((num_layers, num_arrays)) * -1
        self.stats_filter_dram_start_cycl = np.ones((num_layers, num_arrays)) * -1
        self.stats_filter_dram_end_cycl = np.ones((num_layers, num_arrays)) * -1

        self.stats_ofmap_dram_writes = np.ones((num_layers, num_arrays)) * -1
        self.stats_ofmap_dram_start_cycl = np.ones((num_layers, num_arrays)) * -1
        self.stats_ofmap_dram_end_cycl = np.ones((num_layers, num_arrays)) * -1 

        self.total_ifmap_dram_reads = []
        self.total_filter_dram_reads = []
        self.total_ofmap_dram_writes = []

        ########################### SRAM
        self.stats_ifmap_sram_reads = np.ones((num_layers, num_arrays)) * -1
        self.stats_filter_sram_reads = np.ones((num_layers, num_arrays)) * -1
        self.stats_ofmap_sram_writes = np.ones((num_layers, num_arrays)) * -1

        self.total_ifmap_sram_reads = []
        self.total_filter_sram_reads = []
        self.total_ofmap_sram_writes = []
        ##############################################

        self.overall_compute_cycles_per_layers = []
        self.overall_util_perc_per_layer = []

        self.overall_compute_cycles_all_layers = 0
        self.overall_util_perc_all_layer = 0

        self.dataflow = dataflow 
        self.params_valid = True

    #
    def run_simulation_single_layer(self, layer_id=0):

        # Blank 2. Create the operand matrices
        opmat_obj = opmat() 
        opmat_obj.set_params(config_obj=self.single_arr_cfg, topoutil_obj=self.topo_obj, layer_id=layer_id) 

        _, ifmap_op_mat = opmat_obj.get_ifmap_matrix()
        _, filter_op_mat = opmat_obj.get_filter_matrix()
        _, ofmap_op_mat = opmat_obj.get_ofmap_matrix() 
        # <Insert code here>

        ### For GET UTILIZATION ###########################################
        SA_row, SA_col = self.single_arr_cfg.get_array_dims()
        SR=ifmap_op_mat.shape[0]
        SC=filter_op_mat.shape[1]
        #util_this_layer = Get_Util(SA_row, SA_col, SR, SC)
        #self.OVER_util.append(util_this_layer)
        ###################################################################

        ### For OPTIMAL POD DIMENSION #####################################
        optpod = GetOptPodDim(self.num_pods, SA_row, SA_col, SR, SC)
        self.grid_rows, self.grid_cols = map(int, optpod.split('x'))
        print("Pod dimension: {}x{}".format(self.grid_rows, self.grid_cols))
        ##################################################################

        ### For new tiling strategy #######################################
        SA_row, SA_col = self.single_arr_cfg.get_array_dims()
        if_row = ifmap_op_mat.shape[0]
        fil_col = filter_op_mat.shape[1]
        #print(if_row)

        #for grid_row
        num_if_tiles = int(np.ceil(if_row/SA_row)) # compute number of IFMAP tiles
        Used_row = []
        for i in range(self.grid_rows): 
            temporal = np.ceil(num_if_tiles/(i+1)) 
            Used_row.append(temporal) 
        Used_row = np.argmin(Used_row) + 1 

        Tiles_each_pod = []
        self.ifmap_tiles_each_pod_acc = []
        for j in range(Used_row):
            Tiles_each_pod.append(0)
            self.ifmap_tiles_each_pod_acc.append(0)
        
        for k in range(num_if_tiles):
            if (k == num_if_tiles-1)&(if_row%SA_row != 0):
                Tiles_each_pod[k%Used_row] = Tiles_each_pod[k%Used_row]+(if_row % SA_row)
            else:
                Tiles_each_pod[k%Used_row] = Tiles_each_pod[k%Used_row]+SA_row
        
        for l in range(len(Tiles_each_pod)):
            self.ifmap_tiles_each_pod_acc[l] = np.sum(Tiles_each_pod[0:(l+1)])

        #for grid_col
        num_fil_tiles = int(np.ceil(fil_col/SA_col))
        Used_col = []
        for i in range(self.grid_cols):
            temporal = np.ceil(num_fil_tiles/(i+1))
            Used_col.append(temporal)
        Used_col = np.argmin(Used_col) + 1

        Tiles_each_pod = []
        self.filter_tiles_each_pod_acc = []
        for j in range(Used_col):
            Tiles_each_pod.append(0)
            self.filter_tiles_each_pod_acc.append(0)
        
        for k in range(num_fil_tiles):
            if (k == num_fil_tiles-1)&(fil_col%SA_col != 0):
                Tiles_each_pod[k%Used_col] = Tiles_each_pod[k%Used_col]+(fil_col % SA_col)
            else:
                Tiles_each_pod[k%Used_col] = Tiles_each_pod[k%Used_col]+SA_col
        
        for l in range(len(Tiles_each_pod)):
            self.filter_tiles_each_pod_acc[l] = np.sum(Tiles_each_pod[0:(l+1)])
        
        ### pod utilization #########################
        #print("POD UTIL DEBUG: {} {} {} {}".format(Used_row, Used_col, self.grid_rows, self.grid_cols))
        pod_util_this_layer = (Used_row * Used_col) / (self.grid_rows * self.grid_cols)
        assert pod_util_this_layer <= 1.0, "pod is bigger than 1.0"
        self.POD_util.append(pod_util_this_layer) # 

        #DEBUG
        #print('DEBUG: {} {}'.format(self.ifmap_tiles_each_pod_acc, self.filter_tiles_each_pod_acc))
        #print('DEBUG: {} {}'.format(Used_row, Used_col))
        # print(3/0)
        ###########################################################

        PE_util_this_layer=[]

        for grid_row_id in range(self.grid_rows):
            for grid_col_id in range(self.grid_cols): 

#################################################################
                if (grid_row_id >= Used_row) | (grid_col_id >= Used_col):
                    continue
#################################################################
                
                arr_id = grid_row_id * self.grid_cols + grid_col_id 
                print('Running subarray ' + str(arr_id))

                ifmap_op_mat_part, filter_op_mat_part, ofmap_op_mat_part =\
                    self.get_opmat_parts(ifmap_op_mat, filter_op_mat, ofmap_op_mat,
                                         grid_row_id, grid_col_id)
###############################################################
                if ifmap_op_mat_part.shape[0]*ifmap_op_mat_part.shape[1]==0 or\
                    filter_op_mat_part.shape[0]*filter_op_mat_part.shape[1]==0 or\
                        ofmap_op_mat_part.shape[0]*ofmap_op_mat_part.shape[1]==0:
                        continue
###############################################################

                #print('ifmap: {}'.format(ifmap_op_mat_part.shape))
                #print('filter: {}'.format(filter_op_mat_part.shape))

                ### PE util #########################
                Util_this_pod = Get_Util(SA_row, SA_col, ifmap_op_mat_part.shape[0], filter_op_mat_part.shape[1])
                assert Util_this_pod <= 1.0, "PE util in this pod is bigger than 1.0"
                PE_util_this_layer.append(Util_this_pod)
                #######################################################

                # Blank 3. Instantiate the mapping utilities
                compute_system = systolic_compute_os()
                if self.dataflow == 'ws':
                    compute_system = systolic_compute_ws()
                elif self.dataflow == 'is':
                    compute_system = systolic_compute_is() 
                compute_system.set_params(config_obj=self.single_arr_cfg,
                                        ifmap_op_mat=ifmap_op_mat_part,
                                        filter_op_mat=filter_op_mat_part,
                                        ofmap_op_mat=ofmap_op_mat_part) 


                #############################################
                ifmap_prefetch_mat, filter_prefetch_mat = compute_system.get_prefetch_matrices()
                #print('ifmap_prefetch_mat: {}'.format(ifmap_prefetch_mat.shape))
                #############################################


                ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat = compute_system.get_demand_matrices()
                #print('*ifmap_part: {}'.format(ifmap_op_mat_part.shape))
                #print('ifmap_part[-1]: {}'.format(ifmap_op_mat_part[-1]))
                #print('*ifmap shape: {}'.format(ifmap_demand_mat.shape)) ######
                #print(np.max(ifmap_demand_mat))
                #print('*filter shape: {}'.format(filter_demand_mat.shape))
                #print('*ofmap shape: {}'.format(ofmap_demand_mat.shape))
                #<Insert code here>

                # Blank 4. Memory system
                memory_system = mem_dbsp()

                
                ifmap_buf_size_kb, filter_buf_size_kb, ofmap_buf_size_kb = self.single_arr_cfg.get_mem_sizes()
                ifmap_buf_size_bytes = ifmap_buf_size_kb
                filter_buf_size_bytes = filter_buf_size_kb
                ofmap_buf_size_bytes = ofmap_buf_size_kb

                arr_row, arr_col = self.single_arr_cfg.get_array_dims()

                ifmap_backing_bw = 1
                filter_backing_bw = 1
                ofmap_backing_bw = 1


                ###################################
                estimate_bandwidth_mode = False
                if self.single_arr_cfg.use_user_dram_bandwidth():
                    # bws = self.single_arr_cfg.get_bandwidths_as_list()
                    # print('BWs: {}'.format(bws))
                    # ifmap_backing_bw = bws[0]
                    # filter_backing_bw = bws[0]
                    # ofmap_backing_bw = bws[0]
                    arr_row, arr_col = self.single_arr_cfg.get_array_dims()

                    # The number 10 elems per cycle is arbitrary
                    if self.dataflow == 'os' or self.dataflow == 'ws':
                        ifmap_backing_bw = arr_row
                        filter_backing_bw = arr_col
                        ofmap_backing_bw = arr_col

                    elif self.dataflow == 'is':
                        ifmap_backing_bw = arr_col
                        filter_backing_bw = arr_row
                        ofmap_backing_bw = arr_col

                # 
                else:
                    dataflow = self.single_arr_cfg.get_dataflow()
                    arr_row, arr_col = self.single_arr_cfg.get_array_dims()
                    estimate_bandwidth_mode = True

                    # The number 10 elems per cycle is arbitrary
                    if self.dataflow == 'os' or self.dataflow == 'ws':
                        ifmap_backing_bw = 10
                        filter_backing_bw = 10
                        ofmap_backing_bw = 10

                    elif self.dataflow == 'is':
                        ifmap_backing_bw = arr_col
                        filter_backing_bw = arr_row
                        ofmap_backing_bw = arr_col
                #########################################



                # if self.dataflow == 'os' or self.dataflow == 'ws':
                #     ifmap_backing_bw = arr_row
                #     filter_backing_bw = arr_col
                #     ofmap_backing_bw = arr_col

                # elif self.dataflow == 'is':
                #     ifmap_backing_bw = arr_col
                #     filter_backing_bw = arr_row
                #     ofmap_backing_bw = arr_col
                #print('bw: {}'.format(ifmap_backing_bw))
                memory_system.set_params(
                    word_size=1,
                    ifmap_buf_size_bytes=ifmap_buf_size_bytes,
                    filter_buf_size_bytes=filter_buf_size_bytes,
                    ofmap_buf_size_bytes=ofmap_buf_size_bytes,
                    rd_buf_active_frac=0.5, wr_buf_active_frac=0.5,
                    ifmap_backing_buf_bw=ifmap_backing_bw,
                    filter_backing_buf_bw=filter_backing_bw,
                    ofmap_backing_buf_bw=ofmap_backing_bw,
                    verbose=True,
                    estimate_bandwidth_mode=estimate_bandwidth_mode#
                )

                
                ###################################
                if self.single_arr_cfg.use_user_dram_bandwidth() :
                    memory_system.set_read_buf_prefetch_matrices(ifmap_prefetch_mat=ifmap_prefetch_mat,
                                                                    filter_prefetch_mat=filter_prefetch_mat)
                #####################################


                memory_system.service_memory_requests(ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat)
                #<Insert code here>
                
                idram = memory_system.get_ifmap_dram_details()
                #print('IFMAP DRAM read in this grid: {}'.format(idram[2]))
                fildram = memory_system.get_filter_dram_details()
                #print('filter DRAM read in this grid: {}'.format(fildram[2]))
                self.ofdram = memory_system.get_ofmap_dram_details()

                ### WS dram write debug ######################################
                if self.dataflow=='ws':
                    IF_folds=np.ceil(ifmap_op_mat_part.shape[1]/SA_row)
                    #Fil_folds=np.ceil(filter_op_mat_part.shape[1]/SA_col)
                    num_folds=IF_folds
                    self.ofdram=np.ceil(self.ofdram[2]/num_folds)
                    ######################################################################
                    #print('OFMAP DRAM writes in this grid: {}'.format(self.ofdram))
                    #print('DRAMs= {}, {}, {}'.format(idram[2],fildram[2],ofdram[2]))

                self.gather_stats(row_id=grid_row_id,
                                  col_id=grid_col_id,
                                  memory_system_obj=memory_system,
                                  compute_system_obj=compute_system,
                                  layer_id=layer_id)

        assert len(PE_util_this_layer)==(Used_row * Used_col), "util is not computed correctly."
        PE_util_this_layer = np.average(PE_util_this_layer) 
        assert PE_util_this_layer <= 1.0, "PE util is bigger than 1.0"
        Entire_util_this_layer = pod_util_this_layer * PE_util_this_layer 
        print("UTIL DEBUG: {} {} {}".format(pod_util_this_layer, PE_util_this_layer, Entire_util_this_layer))
        assert Entire_util_this_layer <= 1.0, "util is bigger than 1.0"

        self.OVER_util.append(Entire_util_this_layer)

        self.all_grids_done = True

    #
    def run_simulations_all_layers(self):
        assert self.params_valid, 'Params are not valid'

        for lid in range(self.topo_obj.get_num_layers()):
            print('Running layer=' + str(lid))
            self.run_simulation_single_layer(lid)

    #
    def get_opmat_parts(self, ifmap_op_mat, filter_op_mat, ofmap_op_mat,
                        grid_row_id, grid_col_id):
        
        ifmap_op_mat_part = np.zeros((1,1))
        filter_op_mat_part = np.zeros((1,1))
        ofmap_op_mat_part = np.zeros((1,1))
        if self.dataflow == 'os':
            #print('ifmap_row and col: {} {}, filter row and col: {} {}'.format(ifmap_op_mat.shape[0], ifmap_op_mat.shape[1], filter_op_mat.shape[0], filter_op_mat.shape[1]))
            if grid_row_id == 0:
                ifmap_row_start_id = 0
            else:
                ifmap_row_start_id = self.ifmap_tiles_each_pod_acc[grid_row_id-1]
            ifmap_row_end_id = self.ifmap_tiles_each_pod_acc[grid_row_id]
            ifmap_op_mat_part = ifmap_op_mat[ifmap_row_start_id:ifmap_row_end_id, :]

            if grid_col_id == 0:
                filter_col_start_id = 0
            else:
                filter_col_start_id = self.filter_tiles_each_pod_acc[grid_col_id-1]
            filter_col_end_id = self.filter_tiles_each_pod_acc[grid_col_id]
            filter_op_mat_part = filter_op_mat[:, filter_col_start_id:filter_col_end_id]

            if grid_row_id == 0:
                ofmap_row_start_id = 0
            else:
                ofmap_row_start_id = self.ifmap_tiles_each_pod_acc[grid_row_id-1]
            ofmap_row_end_id = self.ifmap_tiles_each_pod_acc[grid_row_id]

            if grid_col_id == 0:
                ofmap_col_start_id = 0
            else:
                ofmap_col_start_id = self.filter_tiles_each_pod_acc[grid_col_id-1]
            ofmap_col_end_id = self.filter_tiles_each_pod_acc[grid_col_id]
            ofmap_op_mat_part = ofmap_op_mat[ofmap_row_start_id: ofmap_row_end_id,
                                             ofmap_col_start_id: ofmap_col_end_id]

        # WS tiling debug ///DONE
        elif self.dataflow == 'ws':
            #print('ifmap_row and col: {} {}, filter row and col: {} {}'.format(ifmap_op_mat.shape[0], ifmap_op_mat.shape[1], filter_op_mat.shape[0], filter_op_mat.shape[1]))
            if grid_row_id == 0:
                ifmap_row_start_id = 0
            else:
                ifmap_row_start_id = self.ifmap_tiles_each_pod_acc[grid_row_id-1]
            ifmap_row_end_id = self.ifmap_tiles_each_pod_acc[grid_row_id]
            ifmap_op_mat_part = ifmap_op_mat[ifmap_row_start_id:ifmap_row_end_id, :]

            if grid_col_id == 0:
                filter_col_start_id = 0
            else:
                filter_col_start_id = self.filter_tiles_each_pod_acc[grid_col_id-1]
            filter_col_end_id = self.filter_tiles_each_pod_acc[grid_col_id]
            filter_op_mat_part = filter_op_mat[:, filter_col_start_id:filter_col_end_id]

            if grid_row_id == 0:
                ofmap_row_start_id = 0
            else:
                ofmap_row_start_id = self.ifmap_tiles_each_pod_acc[grid_row_id-1]
            ofmap_row_end_id = self.ifmap_tiles_each_pod_acc[grid_row_id]

            if grid_col_id == 0:
                ofmap_col_start_id = 0
            else:
                ofmap_col_start_id = self.filter_tiles_each_pod_acc[grid_col_id-1]
            ofmap_col_end_id = self.filter_tiles_each_pod_acc[grid_col_id]
            ofmap_op_mat_part = ofmap_op_mat[ofmap_row_start_id: ofmap_row_end_id,
                                             ofmap_col_start_id: ofmap_col_end_id]

        # IS
        elif self.dataflow == 'is':
            ifmap_rows_per_part = math.ceil(ifmap_op_mat.shape[0] / self.grid_rows)
            ifmap_row_start_id = grid_row_id * ifmap_rows_per_part
            ifmap_row_end_id = min(ifmap_row_start_id + ifmap_rows_per_part, ifmap_op_mat.shape[0])

            ifmap_cols_per_part = math.ceil(ifmap_op_mat.shape[1] / self.grid_cols)
            ifmap_col_start_id = grid_col_id * ifmap_cols_per_part
            ifmap_col_end_id = min(ifmap_col_start_id + ifmap_cols_per_part, ifmap_op_mat.shape[1])
            ifmap_op_mat_part = ifmap_op_mat[ifmap_row_start_id:ifmap_row_end_id,
                                             ifmap_col_start_id:ifmap_col_end_id]

            filter_rows_per_part = math.ceil(filter_op_mat.shape[0] / self.grid_cols)
            filter_row_start_id = grid_col_id * filter_rows_per_part
            filter_row_end_id = min(filter_row_start_id + filter_rows_per_part, filter_op_mat.shape[0])

            filter_op_mat_part = filter_op_mat[filter_row_start_id:filter_row_end_id,:]

            ofmap_rows_per_part = math.ceil(ofmap_op_mat.shape[0] / self.grid_rows)
            ofmap_row_start_id = grid_row_id * ofmap_rows_per_part
            ofmap_row_end_id = min(ofmap_row_start_id + ofmap_rows_per_part, ofmap_op_mat.shape[0])

            ofmap_op_mat_part = ofmap_op_mat[ofmap_row_start_id: ofmap_row_end_id, :]

        return ifmap_op_mat_part, filter_op_mat_part, ofmap_op_mat_part

    #
    def gather_stats(self, memory_system_obj, compute_system_obj,row_id, col_id, layer_id):
        # Stats to gather
        indx = row_id * self.grid_cols + col_id

        # 1. Compute cycles
        self.stats_compute_cycles[layer_id, indx] = memory_system_obj.get_total_compute_cycles()

        # 2. Bandwidth requirements
        ifmap_start_cycle, ifmap_end_cycle, ifmap_dram_reads = memory_system_obj.get_ifmap_dram_details()
        filter_start_cycle, filter_end_cycle, filter_dram_reads = memory_system_obj.get_filter_dram_details()
        ofmap_start_cycle, ofmap_end_cycle, ofmap_dram_writes = memory_system_obj.get_ofmap_dram_details()
        if self.dataflow=='ws':
            ofmap_dram_writes = self.ofdram # 

        ############## SRAM
        ifmap_sram_reads = compute_system_obj.get_ifmap_requests()
        filter_sram_reads = compute_system_obj.get_filter_requests()
        ofmap_sram_writes = compute_system_obj.get_ofmap_requests()

        self.stats_ifmap_sram_reads[layer_id, indx] = ifmap_sram_reads
        self.stats_filter_sram_reads[layer_id, indx] = filter_sram_reads
        self.stats_ofmap_sram_writes[layer_id, indx] = ofmap_sram_writes
        #########################################################

        self.stats_ifmap_dram_reads[layer_id, indx] = ifmap_dram_reads
        self.stats_filter_dram_reads[layer_id, indx] = filter_dram_reads
        self.stats_ofmap_dram_writes[layer_id, indx] = ofmap_dram_writes

        self.stats_ifmap_dram_start_cycl[layer_id, indx] = ifmap_start_cycle
        self.stats_filter_dram_start_cycl[layer_id, indx] = filter_start_cycle
        self.stats_ofmap_dram_start_cycl[layer_id, indx] = ofmap_start_cycle

        self.stats_ifmap_dram_end_cycl[layer_id, indx] = ifmap_end_cycle
        self.stats_filter_dram_end_cycl[layer_id, indx] = filter_end_cycle
        self.stats_ofmap_dram_end_cycl[layer_id, indx] = ofmap_end_cycle

    #
    def calc_overall_stats_all_layer(self):
        assert self.all_grids_done, 'Not all data is available'

        num_layers = self.topo_obj.get_num_layers()
        for layer_id in range(num_layers):
            # 1. Compute cycles
            this_layer_compute_cycles = max(self.stats_compute_cycles[layer_id])
            self.overall_compute_cycles_per_layers += [this_layer_compute_cycles]

            # 2. Overall utilization
            num_compute = self.topo_obj.get_layer_num_ofmap_px(layer_id=layer_id) \
                          * self.topo_obj.get_layer_window_size(layer_id=layer_id)

            row, col = self.single_arr_cfg.get_array_dims()
            total_compute_possible = self.grid_cols * self.grid_rows * row * col * this_layer_compute_cycles
            this_layer_overall_util_perc = num_compute / total_compute_possible * 100

            self.overall_util_perc_per_layer += [this_layer_overall_util_perc]

            # 3. Memory stats
            self.total_ifmap_dram_reads += [sum(self.stats_ifmap_dram_reads[layer_id])]
            self.total_filter_dram_reads += [sum(self.stats_filter_dram_reads[layer_id])]
            self.total_ofmap_dram_writes += [sum(self.stats_ofmap_dram_writes[layer_id])]

            ################## SRAM
            self.total_ifmap_sram_reads += [sum(self.stats_ifmap_sram_reads[layer_id])]
            self.total_filter_sram_reads += [sum(self.stats_filter_sram_reads[layer_id])]
            self.total_ofmap_sram_writes += [sum(self.stats_ofmap_sram_writes[layer_id])]
            #################################################

        self.overall_compute_cycles_all_layers = sum(self.overall_compute_cycles_per_layers)
        self.overall_util_perc_all_layer = sum(self.overall_util_perc_per_layer) / num_layers

        self.metrics_ready = True

    #
    def get_report_items(self):
        return self.overall_compute_cycles_all_layers, self.overall_util_perc_all_layer, \
               self.total_ifmap_dram_reads, self.total_filter_dram_reads, self.total_ofmap_dram_writes, \
               self.total_ifmap_sram_reads, self.total_filter_sram_reads, self.total_ofmap_sram_writes,self.overall_compute_cycles_per_layers, \
               self.OVER_util, self.POD_util
               

def Get_Util(SA_row, SA_col, SR, SC):
    #Fold
    FR=int(np.ceil(SR/SA_row))
    FC=int(np.ceil(SC/SA_col))
    
    Util_row = np.zeros(FR)
    for i in range(FR):
        if (i == (FR-1)) & ((SR % SA_row) != 0):
            Util_row[i] = SR % SA_row
        else:
            Util_row[i] = SA_row
    
    Util_col = np.zeros(FC)
    for i in range(FC):
        if (i == (FC-1)) & ((SC % SA_col) != 0):
            Util_col[i] = SC % SA_col
        else:
            Util_col[i] = SA_col

    Utilization = np.zeros([FR, FC])
    for i in range(FR):
        for j in range(FC):
            Utilization[i][j] = (Util_row[i] * Util_col[j]) / (SA_row * SA_col)
    Utilization = np.average(Utilization)

    return Utilization

def GetOptPodDim(num_pod, SA_row, SA_col, if_row, fil_col): 
### For Get Optimal Pod Dimension #################################
    i = 1
    pod_dict = {}
    pod_row = []
    while (i<=num_pod):
        pod_row.append(i)
        pod_dict[int(np.log2(i))] = str(i) + 'x' + str(int(num_pod/i))
        i = i * 2
    print(pod_dict)
    print(pod_row)
    # pod_dict={0:'1x256', 1:'2x128', 2:'4x64', 3:'8x32', 4:'16x16', 5:'32x8', 6:'64x4', 7:'128x2', 8:'256x1'}
    # pod_row = [1,2,4,8,16,32,64,128,256]
    for_runtime = [] #
    for_dram = [] 

    for Pr in pod_row:
        Pc = int(num_pod/Pr) # 1x16, 2x8, 4x4

        # for grid_row
        num_if_tiles = int(np.ceil(if_row/SA_row)) # compute number of IFMAP tiles
        Used_row = []
        for i in range(Pr): #
            temporal = np.ceil(num_if_tiles/(i+1)) 
            Used_row.append(temporal) 
        if_tile_per_pod = np.min(Used_row)
        Used_row = np.argmin(Used_row) + 1 
        
        #for grid_col
        num_fil_tiles = int(np.ceil(fil_col/SA_col))
        Used_col = []
        for i in range(Pc):
            temporal = np.ceil(num_fil_tiles/(i+1))
            Used_col.append(temporal)
        fil_tile_per_pod = np.min(Used_col)
        Used_col = np.argmin(Used_col) + 1 

        rt_this_dim = if_tile_per_pod * fil_tile_per_pod

        #
        base_if_tiles = (num_if_tiles - (num_if_tiles % Used_row))/Used_row
        pod_if_tiles = [] 
        for i in range(Used_row):
            if (i+1) <= (num_if_tiles % Used_row):
                pod_if_tiles.append(base_if_tiles + 1)
            else:
                pod_if_tiles.append(base_if_tiles)

        base_fil_tiles = (num_fil_tiles - (num_fil_tiles % Used_col))/Used_col
        pod_fil_tiles = [] 
        for i in range(Used_col):
            if (i+1) <= (num_fil_tiles % Used_col):
                pod_fil_tiles.append(base_fil_tiles + 1)
            else:
                pod_fil_tiles.append(base_fil_tiles)
        
        dram_this_dim = 0
        for i in pod_if_tiles:
            for j in pod_fil_tiles:
                temp = i+j
                dram_this_dim = dram_this_dim + temp

        for_runtime.append(rt_this_dim)
        for_dram.append(dram_this_dim)
    
    runtime_min_pods = []
    for i in range(len(for_runtime)):
        print('{}: rt={} dram={}'.format(pod_dict[i], for_runtime[i], for_dram[i]))
        if for_runtime[i]==np.min(for_runtime):
            runtime_min_pods.append(i)
    if len(runtime_min_pods)==1:
        opt_pod_dim=runtime_min_pods[0]
    else:
        runtime_min_drams=[]
        dram_min_pods=[]
        for j in runtime_min_pods:
            runtime_min_drams.append(for_dram[j])
        for j in runtime_min_pods:
            if for_dram[j]==np.min(runtime_min_drams):
                dram_min_pods.append(j)
        opt_pod_dim=dram_min_pods[0]
    print("opt_pod_dim={}".format(pod_dict[opt_pod_dim]))

    return pod_dict[opt_pod_dim]

def Get_weighted_util(cycles_per_layer, util_per_layer):
    Wutil=[]
    for i in range(len(cycles_per_layer)):
        CxU = (cycles_per_layer[i]/np.sum(cycles_per_layer)) * util_per_layer[i] * 100 
        Wutil.append(CxU)
    Wutil = np.round(np.sum(Wutil),2)

    return Wutil

def Get_OFMAP_offchip_writes(toponame):
    topofile = './topologies/conv_nets/'+toponame+'.csv'
    csvtopo=open(topofile,'r',encoding='cp949')

    ifparam=[]
    filparam=[]
    ofparam=[]

    csvtopo=open(topofile,'r',encoding='cp949')
    toporow=csv.reader(csvtopo)
    next(toporow)
    for topo in toporow:
        I_row=int(topo[1])
        I_col=int(topo[2])
        F_row=int(topo[3])
        F_col=int(topo[4])
        S=int(topo[7])
        Chan=int(topo[5])
        numFil=int(topo[6])

        ifparam_this_layer = I_row * I_col * Chan
        filparam_this_layer = F_row * F_col * Chan * numFil
        ofparam_this_layer = np.ceil( (I_row-F_row+S)/S ) * np.ceil( (I_col-F_col+S)/S )
        ofparam_this_layer = ofparam_this_layer * numFil
        ifparam.append(ifparam_this_layer)
        filparam.append(filparam_this_layer)
        ofparam.append(ofparam_this_layer)

    return ofparam

#



#
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('num_pod', help='Set the number of pods')
    parser.add_argument('topo', help='Set topology file')
    args = parser.parse_args()
    ##################################
    # SIM CONFIGURE #
    ##################################
    simname= 'NewTile8x8' 
    toponame= args.topo 
    plotname_runtime = 'RT.png' 
    plotname_DRAM = 'DRAM.png' 
    mnk_flag = False 
    
    dflow = 'ws'
    ###########################################################

    topofile = './topologies/conv_nets/'+toponame+'.csv' 
    config_base = './configs/'

    ### Set Config file ###
    createDirectory(config_base)
    cfgfname = "SOSA_16SA"
    cfgname = config_base + cfgfname +'.cfg'
    grid_arr_dict = {} 
    pod_config = args.num_pod + 'x1'
    grid_arr_dict[pod_config] = cfgfname
    #############################
    
    repname = "SOSA_" + "16x16SA_" + args.num_pod
    repdir='./TDP/SOSArep_detail/'+toponame #./reports_detail/simname_toponame
    createDirectory('./SOSArep_detail')
    createDirectory(repdir)
    rep = open(repdir + '/' + repname + '.txt','w') #./reports/toponame/Detail/repname.txt
    
    x_labels = list(grid_arr_dict.keys())
    DRAMS=[]
    SRAMS=[]
    RT=[]

    for gds in x_labels:
        time_start=datetime.now()
        print(time_start)

        print('Running {}...'.format(gds))
        g_row, g_col = map(int, gds.split('x'))

        arrsize = grid_arr_dict[gds]
        config_file = config_base + arrsize + '.cfg'
        grid1 = scaled_out_simulator()
        grid1.set_params(topology_filename=topofile,
                        single_arr_config_file=config_file,
                        grid_rows=g_row, grid_cols=g_col, dataflow=dflow, mnk_input=mnk_flag, num_pods=int(args.num_pod))

        grid1.run_simulations_all_layers()
        grid1.calc_overall_stats_all_layer()

        cycles, util, ifmap_read, filter_reads, ofmap_writes, ifmap_read_sram, filter_reads_sram, ofmap_writes_sram, cycles_per_layer, util_all_layer, pod_util_per_layer = grid1.get_report_items()
                
        Wutil = Get_weighted_util(cycles_per_layer, util_all_layer)
        pod_Wutil = Get_weighted_util(cycles_per_layer, pod_util_per_layer)
        util_all_layer=np.round(np.average(np.array(util_all_layer))*100,2)

        #ofmap_writes = Get_OFMAP_offchip_writes(toponame)
        overall_read=ifmap_read+filter_reads+ofmap_writes
        overall_read_sram=ifmap_read_sram+filter_reads_sram+ofmap_writes_sram
        
        if gds==x_labels[0]:
            all_cycles = [cycles]
            all_utils = [util]
            dram_arr = [[sum(ifmap_read), sum(filter_reads), sum(ofmap_writes)]] 
            sram_arr = [[sum(ifmap_read_sram), sum(filter_reads_sram), sum(ofmap_writes_sram)]] 
            ov_dram = [overall_read]
            ov_sram = [overall_read_sram]
        else:
            all_cycles += [cycles]
            all_utils += [util]
            dram_arr += [[sum(ifmap_read), sum(filter_reads), sum(ofmap_writes)]] 
            sram_arr += [[sum(ifmap_read_sram), sum(filter_reads_sram), sum(ofmap_writes_sram)]] 
            ov_dram += [overall_read]
            ov_sram += [overall_read_sram]

        DRAMS.append(sum(ifmap_read)+sum(filter_reads)+sum(ofmap_writes))
        SRAMS.append(sum(ifmap_read_sram)+sum(filter_reads_sram)+sum(ofmap_writes_sram))
        RT.append(cycles)

        rep.write('\n***{} REPORT\n*Runtime= {}\n*DRAM reads= {}, {}, {}\n*total DRAM reads= {}\n'
            .format(gds, cycles, sum(ifmap_read), sum(filter_reads), sum(ofmap_writes), sum(ifmap_read)+sum(filter_reads)+sum(ofmap_writes)))
        rep.write('*SRAM reads= {}, {}, {}\n*total SRAM reads= {}\n'
            .format(sum(ifmap_read_sram), sum(filter_reads_sram), sum(ofmap_writes_sram), sum(ifmap_read_sram)+sum(filter_reads_sram)+sum(ofmap_writes_sram)))
        rep.write('*Weighted Util:{}\n'.format(Wutil))
        rep.write('*Weighted Pod Util:{}\n'.format(pod_Wutil))
        rep.write('*Per-layer info:\nRuntimes={}\nIFMAP_DRAM={}\nFilter_DRAM={}\nOFMAP_DRAM={}\n'.format(cycles_per_layer, ifmap_read, filter_reads, ofmap_writes))
        rep.write('IFMAP_SRAM={}\nFilter_SRAM={}\nOFMAP_SRAM={}\n'.format(ifmap_read_sram, filter_reads_sram, ofmap_writes_sram))


    print('Overall runtime cycles:{}'.format(sum(all_cycles))) 
    print('Overall DRAM access:{}'.format(sum(ov_dram[0])))
    print('Overall SRAM access:{}'.format(sum(ov_sram[0]))) 
    print('Average PE Utilization:{}'.format(util_all_layer)) 
    print('Compute Util:{}'.format(np.round(util,2)))
    print('*Weighted Util:{}\n'.format(Wutil)) #Wutil
    print('*Weighted Pod Util:{}\n'.format(pod_Wutil)) #pod Wutil

    time_end = datetime.now()
    print(time_end)
    print(time_end - time_start)

    rep.write('\nRT={}'.format(RT))
    rep.write('\nDRAMS={}'.format(DRAMS))
    rep.write('\nSRAMS={}\n'.format(SRAMS))
    rep.write('\nElapsed Time={}\n'.format(str(time_end - time_start)))
    rep.close()

    repdir2='./TDP/SOSArep_compact/'+toponame #./reports_compact/simname_toponame
    createDirectory('./SOSArep_compact')
    createDirectory(repdir2)
    rep2 = open(repdir2 + '/' + repname + '.txt','w') #./reports/toponame/Detail/repname.txt

    rep2.write('Overall runtime cycles:{}\n'.format(sum(all_cycles))) 
    rep2.write('Overall DRAM access:{}\n'.format(sum(ov_dram[0]))) 
    rep2.write('Overall SRAM access:{}\n'.format(sum(ov_sram[0]))) 
    rep2.write('Average PE Utilization:{}\n'.format(util_all_layer)) 
    rep2.write('Compute Util:{}\n'.format(np.round(util,2)))
    rep2.write('DRAM reads:{}\n'.format(sum(ifmap_read)+sum(filter_reads)))
    rep2.write('DRAM writes:{}\n'.format(sum(ofmap_writes)))
    rep2.write('SRAM reads:{}\n'.format(sum(ifmap_read_sram)+sum(filter_reads_sram)))
    rep2.write('SRAM writes:{}\n'.format(sum(ofmap_writes_sram)))
    rep2.write('Weighted Util:{}\n'.format(Wutil))
    rep2.write('Weighted Pod Util:{}\n'.format(pod_Wutil))
    rep2.close()