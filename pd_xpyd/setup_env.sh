apt update
apt install git wget curl net-tools sudo iputils-ping etcd  -y

#Mooncake
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
bash dependencies.sh
mkdir build
cd build
cmake ..
make -j
make install


