#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
int get_device_attribute(int attribute, int device_id)
{
    int device, value;
    if (device_id < 0) {
        device = dpct::dev_mgr::instance().current_device_id();
    }
    else {
        device = device_id;
    }
    
    return dpct::dev_mgr::instance().get_device(device).get_local_mem_size();
    //cudaDeviceGetAttribute(&value, static_cast<int>(attribute), device);
    //return value;
}
