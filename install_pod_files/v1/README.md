To deploy a pod with the vllm that we can edit into a pod to run experimetns do:

1- copy the Dockerfile into the startup-time-logs repo
2- build the image
3- push the image to my repo register quay.io
4- deploy the pod using pod-vm.yaml
5- you can connect to the pod using oc exec -it pod/vllm-vm -- /bin/sh
6- inside the pod run this commands:
# uninstall vllm
pip uninstall -y vllm

# install vllm developer mode, I think that is where the vllm source is inside the image
git clone git@github.com:diegocastanibm/vllm.git
VLLM_USE_PRECOMPILED=1 pip install -e vllm

# install startup-time-logs
pip install .

7- Once everything has been installed, we can run the command:
startup_time_logs -f models.txt --in-page-cache YES --model-loader safetensors --no-sleep-wake --no-chat   --debug
