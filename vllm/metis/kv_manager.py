from typing import List, Tuple, Dict
import torch
import time

class Loading_Optimizer:
    def __init__(self, per_chunk_info: List[Dict[str, int]], gpu_waiting_time) -> None:
        self.num_chunks = len(per_chunk_info)
        self.per_chunk_info = per_chunk_info
        self.gpu_waiting_time = gpu_waiting_time
    
    #for each token chunk, output number of proportion needed and compute the rest.
    #we could also do a loading graph
    def get_loading_plan(self) -> List[Tuple[int, int, int]]: 
        ret = []
        
        for chunk in self.per_chunk_info:
            prefetch_proportion = 0

            #Prefetch plan. TODO what if the other process is using the same bw.
            if (chunk['tier'] > 2):
                time_transmission = chunk['size'] / chunk['bw']
                if (time_transmission > self.gpu_waiting_time):
                    ret.append[1, 0, 0]
                else:
                    prefetch_proportion = self.gpu_waiting_time / time_transmission
                    chunk['size'] -= prefetch_proportion * chunk['bw']
            
            #Now we calculate the pipeline.
                        
        return ret
        

    

class KV_manager:
    def __init__(self, 
                 path:str , 
                 tiers: List[str],
                 attention_layer_cnt: int
                 ) -> None:
        self.path = path
        self.tiers = tiers
        self.cpu_hash = {}
        self.cpu_pin_hash = {}
        self.gpu_hash = {} #key, [size (GB), content]

        #actually load them
        self.disk_hash = {'kv_temp': [233, "/local/hanchen/kv_temp"], 'kv_temp3': [233, "/local/hanchen/kv_temp2"]}
        self.attention_layer_cnt = attention_layer_cnt

        self.loading_opt = Loading_Optimizer([], 0)
        #prepare all kv caches
    
    def query_generation(self):
        self.plan = self.loading_opt.get_loading_plan()

    #key_or_value 0:key 1:value
    def fetch_all_kv_layer(self, text_hash:str, 
                       layer:int, 
                       is_key: bool, 
                       device:str, 
                       mask=None):
        
        start = time.time()
        temp = torch.load(f"{self.disk_hash[text_hash][1]}/{str(layer)}_{int(is_key)}", map_location = torch.device('cpu'))
        mid = time.time()
        print("disk to cpu time is: ", mid - start)
        temp = temp.pin_memory()
        timecheck = time.time()
        print("cpu to pinned cpu is: ", timecheck- mid)
        temp = temp.cuda()
        print("cpu to gpu time is: ", time.time()- timecheck)

        #call an async update function to put this into higher tier.         
        return temp
    

        if (text_hash in self.disk_hash):
            temp = torch.load(f"{self.disk_hash[text_hash][1]}/{str(layer)}_{int(is_key)}", map_location = device)
            return temp
        else:
            return -1
    
    def pre_fetch(self, text_hash:str, mask=None):
        return
        # prefetch to cpu based on LRU, LRU 

        # prefetch part of the cpu from self.plan. If cpu memory exceeds, just move to ssd (deprioritized).  





        # if (text_hash in self.disk_hash):
        #     self.cpu_hash[text_hash] = [self.disk_hash[text_hash][0], [[], []]] #size, [[value_layer0, value layer1, ..], [k_layer0, k_layer1]]

        #     for layer in self.attention_layer_cnt:
        #         self.cpu_hash[text_hash][1][0].append(torch.load(f"{self.disk_hash[text_hash][1]}/{str(layer)}_0", map_location = torch.device('cpu')))
        #         self.cpu_hash[text_hash][1][1].append(torch.load(f"{self.disk_hash[text_hash][1]}/{str(layer)}_1", map_location = torch.device('cpu')))
                
        # return 
            


