sudo apt update
sudo apt install -y nvidia-utils-535
sudo add-apt-repository restricted
sudo add-apt-repository universe
sudo apt update
sudo apt install -y nvidia-driver-535
sudo nvidia-smi -mig 0
sudo reboot
