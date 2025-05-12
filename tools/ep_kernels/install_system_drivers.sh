set -ex

# prepare workspace directory
WORKSPACE=$1
if [ -z "$WORKSPACE" ]; then
    export WORKSPACE=$(pwd)/ep_kernels_workspace
fi

if [ ! -d "$WORKSPACE" ]; then
    mkdir -p $WORKSPACE
fi

# build and install gdrcopy driver
pushd $WORKSPACE
cd gdrcopy_src
./insmod.sh
# run gdrcopy_copybw to test the installation
$WORKSPACE/gdrcopy_install/bin/gdrcopy_copybw

# turn on IBGDA
echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | tee -a /etc/modprobe.d/nvidia.conf
update-initramfs -u

echo "Please reboot the system to apply the changes"
