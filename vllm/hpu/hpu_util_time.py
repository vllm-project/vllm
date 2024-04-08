import torch
import time
import os
import multiprocessing
import re
import numpy as np
import habana_frameworks.torch as htorch
import pyhlml

class HpuUtilTime:
    def __init__(self, interval):
        self.target = self.query_hl_smi
        self.queue = multiprocessing.Queue(maxsize=1)
        self.manager = multiprocessing.Manager().list()
        self.process = multiprocessing.Process(
            target=self.target,
            args=(self.queue,)
        )
        pyhlml.hlmlInit()
        self.device_count = pyhlml.hlmlDeviceGetCount()
        self.device_time = np.zeros(self.device_count, dtype=np.float)
        self.device_mean_utilization = np.zeros(self.device_count, dtype=np.float)
        self.loop_counter = 0
        self.interval = interval
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        self.process.kill()
        if self.target == self.capture_hl_smi:
            self.postprocess_hl_smi()
        pyhlml.hlmlShutdown()
        self.display()

    def start(self):
        self.process.start()
    
    # 1st Method - paring hl-smi query output
    def query_hl_smi(self, queue):
        while True:
            hl_smi_output = os.popen("hl-smi -Q power.draw,utilization.aip -f csv,noheader,nounits").read()
            devices_utilization = re.findall(r'\, (\d+)', hl_smi_output)
            devices_utilization_numpy = np.array(devices_utilization).astype(int)

            self.loop_counter += 1
            time.sleep(self.interval)
            self.device_time[devices_utilization_numpy > 0] += self.interval
            self.device_mean_utilization += (devices_utilization_numpy - self.device_mean_utilization) / self.loop_counter

            devices_output = (
                self.device_mean_utilization,
                self.device_time,
            )
            self.manager.append(devices_output)

    # 2nd Method - paring full hl-smi output
    def hl_smi(self, queue):
        while True:            
            hl_smi_output = os.popen("hl-smi").read()
            devices_utilization = re.findall(r'\b(\d+)%', hl_smi_output)
            devices_utilization_numpy = np.array(devices_utilization).astype(int)

            self.loop_counter += 1
            time.sleep(self.interval)
            self.device_time[devices_utilization_numpy > 0] += self.interval
            self.device_mean_utilization += (devices_utilization_numpy - self.device_mean_utilization) / self.loop_counter

            devices_output = (
                self.device_mean_utilization,
                self.device_time,
            )
            self.manager.append(devices_output)

    # 3rd Method
    def pyhlml(self, queue):
        while True:
            self.loop_counter += 1

            for device_idx in range(self.device_count):
                device = pyhlml.hlmlDeviceGetHandleByIndex(device_idx)
                device_utilization = pyhlml.hlmlDeviceGetUtilizationRates(device)
                # device_utilization = pyhlml.hlmlDeviceGetPowerManagementDefaultLimit(device)
                # device_utilization = pyhlml.hlmlDeviceGetPowerUsage(device)
                self.device_time[device_idx] += self.interval if device_utilization > 0 else 0
                self.device_mean_utilization[device_idx] += (device_utilization - self.device_mean_utilization[device_idx]) / self.loop_counter
            time.sleep(self.interval)

            devices_output = (
                self.device_mean_utilization,
                self.device_time,
            )
            self.manager.append(devices_output)

    # 4th Method
    def capture_hl_smi(self, queue):
        while True:
            hl_smi_output = os.popen("hl-smi -Q power.draw,utilization.aip -f csv,noheader,nounits").read()
            self.manager.append(hl_smi_output)
            time.sleep(self.interval)

    def postprocess_hl_smi(self):
        devices_output = []
        for hl_smi_output in self.manager:
            devices_utilization = re.findall(r'\, (\d+)', hl_smi_output)
            devices_utilization_numpy = np.array(devices_utilization).astype(int)

            self.loop_counter += 1
            time.sleep(self.interval)
            self.device_time[devices_utilization_numpy > 0] += self.interval
            self.device_mean_utilization += (devices_utilization_numpy - self.device_mean_utilization) / self.loop_counter

            devices_output.append((
                self.device_mean_utilization,
                self.device_time,
            ))
        self.manager = devices_output

    # Display final results
    def display(self):
        if self.manager:
            mean_utilization = np.round(self.manager[-1][0], decimals=2)
            time_utilization = np.round(self.manager[-1][1], decimals=6)
            for device_idx in range(self.device_count):
                print(f'HPU {device_idx} | Mean utilization: {mean_utilization[device_idx]}% | Time utilization: {time_utilization[device_idx]}s')