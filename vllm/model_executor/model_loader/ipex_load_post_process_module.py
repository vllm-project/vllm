import intel_extension_for_pytorch as ipex

# ipex (Both CPU and XPU) will have some optimization on model weight layout 
# to fully leverage hardware potential. Firstly, we want to add cpu quant optimization
# cpu would performs best with a specific weight layout (which is different to cuda device layout), 
# so a repack api should be called.

def process_weights_after_loading(model):
    # todo: for quant model, call ipex repack API. 
    # eg: return ipex.repack_awq(model)
    pass