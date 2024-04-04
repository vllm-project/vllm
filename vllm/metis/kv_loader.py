from typing import List
import torch
import time

class kv_manager:
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
        self.disk_hash = {'kv_temp': [233, "/local/hanchen/kv_temp"], 'kv_temp3': [233, "/local/hanchen/kv_temp2"]}
        self.attention_layer_cnt = attention_layer_cnt
        #prepare all kv caches
    
    #key_or_value 0:key 1:value
    def fetch_kv_layer(self, text_hash:str, 
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
        
        return temp
        if (text_hash in self.disk_hash):
            temp = torch.load(f"{self.disk_hash[text_hash][1]}/{str(layer)}_{int(is_key)}", map_location = device)
            return temp
        else:
            return -1
    
    def pre_fetch(self, text_hash:str, mask=None):
        if (text_hash in self.disk_hash):
            self.cpu_hash[text_hash] = [self.disk_hash[text_hash][0], [[], []]] #size, [[value_layer0, value layer1, ..], [k_layer0, k_layer1]]

            for layer in self.attention_layer_cnt:
                self.cpu_hash[text_hash][1][0].append(torch.load(f"{self.disk_hash[text_hash][1]}/{str(layer)}_0", map_location = torch.device('cpu')))
                self.cpu_hash[text_hash][1][1].append(torch.load(f"{self.disk_hash[text_hash][1]}/{str(layer)}_1", map_location = torch.device('cpu')))
                
        return 
            


