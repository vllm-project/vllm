To deploy a pod with the vllm that we can edit into a pod to run experimetns do:

1- copy the Dockerfile into the startup-time-logs repo
2- build the image
3- push the image to my repo register quay.io
4- deploy the pod using pod-vm.yaml
5- you can connect to the pod using oc exec -it pod/vllm-vm -- /bin/sh
