#include "vmm_pool_client.h"
#include "uds_rpc.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>

static void ccheck(CUresult r, const char* m){ if (r!=CUDA_SUCCESS) throw std::runtime_error(m); }
struct CtxScope { CUcontext ctx; CtxScope(CUcontext c){ cuCtxSetCurrent(c);} ~CtxScope(){} };

VmmPoolClient::VmmPoolClient(const std::string& uds_path, int device)
: uds_(uds_path), dev_(device) {
  cuInit(0);
  ccheck(cuDeviceGet(&cu_dev_, dev_), "cuDeviceGet");
  ccheck(cuDevicePrimaryCtxRetain(&ctx_, cu_dev_), "primaryCtxRetain");
  CtxScope guard(ctx_);
  if (!Hello()) throw std::runtime_error("Hello failed");
}

VmmPoolClient::~VmmPoolClient(){
  CtxScope guard(ctx_);
  if (pool_.base_va) cuMemAddressFree(pool_.base_va, pool_.size);
  if (pool_.handle) cuMemRelease(pool_.handle);
  if (hello_fd_>=0) ::close(hello_fd_);
  cuDevicePrimaryCtxRelease(cu_dev_);
}

bool VmmPoolClient::Hello(){
  CtxScope guard(ctx_);
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd<0) throw std::runtime_error("socket");
  sockaddr_un addr{}; addr.sun_family=AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", uds_.c_str());
  if (connect(fd,(sockaddr*)&addr,sizeof(addr))<0){ ::close(fd); throw std::runtime_error("connect"); }
  int cmd=1; ::write(fd, &cmd, sizeof(cmd));
  int32_t st=0; ::read(fd, &st, sizeof(st)); if (st) {::close(fd); return false;}
  uint64_t meta[2]; ::read(fd, meta, sizeof(meta)); pool_.gran=meta[0]; pool_.size=meta[1];

  // recv FD
  struct msghdr msg{}; struct iovec iov; char ch;
  iov.iov_base=&ch; iov.iov_len=1; msg.msg_iov=&iov; msg.msg_iovlen=1;
  char cmsgbuf[CMSG_SPACE(sizeof(int))];
  msg.msg_control=cmsgbuf; msg.msg_controllen=sizeof(cmsgbuf);
  ssize_t n = recvmsg(fd, &msg, 0);
  if (n<=0) {::close(fd); throw std::runtime_error("recvmsg");}
  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg || cmsg->cmsg_len != CMSG_LEN(sizeof(int))) {::close(fd); throw std::runtime_error("bad cmsg");}
  int sfd = *(int*)CMSG_DATA(cmsg);
  ::close(fd);
  hello_fd_ = sfd;

  ccheck(cuMemImportFromShareableHandle(&pool_.handle, &sfd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), "import");
  ccheck(cuMemAddressReserve(&pool_.base_va, pool_.size, 0, 0, 0), "reserve");
  pool_.device = dev_;
  return true;
}

std::pair<uint64_t,uint64_t> VmmPoolClient::Allocate(uint64_t nbytes){
  RpcClient c(uds_);
  int cmd=2; c.send_all(&cmd, sizeof(cmd));
  c.send_all(&nbytes, sizeof(nbytes));
  int32_t st=0; c.recv_all(&st, sizeof(st));
  std::pair<uint64_t,uint64_t> p{~0ull,0};
  c.recv_all(&p, sizeof(p));
  return (st==0)?p:std::pair<uint64_t,uint64_t>{~0ull,0};
}

bool VmmPoolClient::Free(uint64_t off, uint64_t len){
  RpcClient c(uds_);
  int cmd=3; c.send_all(&cmd, sizeof(cmd));
  struct { uint64_t off,len; } s{off,len};
  c.send_all(&s, sizeof(s));
  int32_t st=0; c.recv_all(&st, sizeof(st));
  return st==0;
}

std::tuple<uint64_t,uint64_t,uint64_t> VmmPoolClient::Stats(){
  RpcClient c(uds_);
  int cmd=4; c.send_all(&cmd, sizeof(cmd));
  int32_t st=0; c.recv_all(&st, sizeof(st));
  uint64_t used=0, tot=0, gran=0;
  c.recv_all(&used, sizeof(used));
  c.recv_all(&tot, sizeof(tot));
  c.recv_all(&gran, sizeof(gran));
  return {used, tot, gran};
}

CUdeviceptr VmmPoolClient::Map(uint64_t off, uint64_t len){
  CtxScope guard(ctx_);
  uint64_t L = ((len + pool_.gran - 1)/pool_.gran)*pool_.gran;
  CUdeviceptr va = pool_.base_va + off;
  ccheck(cuMemMap(va, L, off, pool_.handle, 0), "map");
  CUmemAccessDesc acc{};
  acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE; acc.location.id = dev_;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ccheck(cuMemSetAccess(va, L, &acc, 1), "setaccess");
  return va;
}

void VmmPoolClient::Unmap(uint64_t off, uint64_t len){
  CtxScope guard(ctx_);
  uint64_t L = ((len + pool_.gran - 1)/pool_.gran)*pool_.gran;
  CUdeviceptr va = pool_.base_va + off;
  cuMemUnmap(va, L);
}
