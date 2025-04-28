# Security Guide

## Inter-Node Communication

All communications between nodes in a multi-node vLLM deployment are **insecure by default** and must be protected by placing the nodes on an isolated network. This includes:

1. PyTorch Distributed communications
2. KV cache transfer communications
3. Tensor, Pipeline, and Data parallel communications

### Configuration Options for Inter-Node Communications

The following options control inter-node communications in vLLM:

1. **Environment Variables:**
   - `VLLM_HOST_IP`: Sets the IP address for vLLM processes to communicate on

2. **KV Cache Transfer Configuration:**
   - `--kv-ip`: The IP address for KV cache transfer communications (default: 127.0.0.1)
   - `--kv-port`: The port for KV cache transfer communications (default: 14579)

3. **Data Parallel Configuration:**
   - `data_parallel_master_ip`: IP of the data parallel master (default: 127.0.0.1)
   - `data_parallel_master_port`: Port of the data parallel master (default: 29500)

### Notes on PyTorch Distributed

vLLM uses PyTorch's distributed features for some inter-node communication. For
detailed information about PyTorch Distributed security considerations, please
refer to the [PyTorch Security
Guide](https://github.com/pytorch/pytorch/security/policy#using-distributed-features).

Key points from the PyTorch security guide:
- PyTorch Distributed features are intended for internal communication only
- They are not built for use in untrusted environments or networks
- No authorization protocol is included for performance reasons
- Messages are sent unencrypted
- Connections are accepted from anywhere without checks

### Security Recommendations

1. **Network Isolation:**
   - Deploy vLLM nodes on a dedicated, isolated network
   - Use network segmentation to prevent unauthorized access
   - Implement appropriate firewall rules

2. **Configuration Best Practices:**
   - Always set `VLLM_HOST_IP` to a specific IP address rather than using defaults
   - Configure firewalls to only allow necessary ports between nodes

3. **Access Control:**
   - Restrict physical and network access to the deployment environment
   - Implement proper authentication and authorization for management interfaces
   - Follow the principle of least privilege for all system components

## Reporting Security Vulnerabilities

If you believe you have found a security vulnerability in vLLM, please report it following the project's security policy. For more information on how to report security issues and the project's security policy, please see the [vLLM Security Policy](https://github.com/vllm-project/vllm/blob/main/SECURITY.md).
