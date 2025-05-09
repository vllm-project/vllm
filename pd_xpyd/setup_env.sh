echo "1. setting up general requirement......."
apt update
apt install git wget curl net-tools sudo iputils-ping etcd  -y

echo "2. setting up mooncake mooncake-transfer-engine==0.3.0b3............."
#Mooncake
pip3 install mooncake-transfer-engine==0.3.0b3

echo "3. setting up RDMA for mooncake ..................."
#RDMA
apt remove ibutils libpmix-aws
wget https://www.mellanox.com/downloads/DOCA/DOCA_v2.10.0/host/doca-host_2.10.0-093000-25.01-ubuntu2204_amd64.deb
dpkg -i doca-host_2.10.0-093000-25.01-ubuntu2204_amd64.deb
apt-get update
apt-get -y install doca-ofed

ibdev2netdev
#mlx5_0 port 1 ==> ens108np0 (Up)
#mlx5_1 port 1 ==> ens9f0np0 (Up)
#mlx5_2 port 1 ==> ens9f1np1 (Up)
#mlx5_3 port 1 ==> ens109np0 (Up)
#mlx5_4 port 1 ==> ens110np0 (Up)
#mlx5_5 port 1 ==> ens111np0 (Up)
#mlx5_6 port 1 ==> ens112np0 (Up)
#mlx5_7 port 1 ==> ens113np0 (Up)
#mlx5_8 port 1 ==> ens114np0 (Up)
#mlx5_9 port 1 ==> ens115np0 (Up)

