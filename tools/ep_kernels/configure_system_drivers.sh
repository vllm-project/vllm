set -ex

# turn on IBGDA
echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | tee -a /etc/modprobe.d/nvidia.conf
update-initramfs -u

echo "Please reboot the system to apply the changes"
