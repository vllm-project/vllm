# install nvidia drivers and tools
sudo apt update
sudo add-apt-repository -y restricted
sudo add-apt-repository -y universe
sudo apt update
sudo apt install -y nvidia-utils-550 nvidia-driver-550
sudo nvidia-smi -mig 0 # disable multi instance GPU
sudo reboot
