#include <cuda.h>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <cstring>
#include <iostream>
#include <algorithm>
#include "uds_rpc.h"

struct Slice { uint64_t off; uint64_t len; };

static CUdevice g_dev;
static CUcontext g_ctx;
static CUmemGenericAllocationHandle g_pool;
static size_t g_gran{2ull<<20};
static uint64_t g_pool_bytes{0};
static int g_share_fd{-1};

static std::mutex g_mtx;
static std::vector<Slice> g_free;
static std::atomic<uint64_t> g_used{0};

static void check(CUresult r, const char* msg){ if (r!=CUDA_SUCCESS){ std::cerr<<msg<<":"<<r<<"\n"; std::exit(2);} }

static void coalesce_sorted(std::vector<Slice>& v){
  if (v.empty()) return;
  std::sort(v.begin(), v.end(), [](const Slice&a, const Slice&b){return a.off<b.off;});
  size_t w=0;
  for (size_t i=1;i<v.size();++i){
    if (v[w].off + v[w].len == v[i].off) v[w].len += v[i].len;
    else v[++w] = v[i];
  }
  v.resize(w+1);
}

static Slice alloc_slice(uint64_t bytes) {
  std::lock_guard<std::mutex> lk(g_mtx);
  uint64_t need = ((bytes + g_gran - 1)/g_gran)*g_gran;
  for (size_t i=0;i<g_free.size();++i){
    if (g_free[i].len >= need) {
      uint64_t off = g_free[i].off;
      g_free[i].off += need;
      g_free[i].len -= need;
      if (g_free[i].len == 0) g_free.erase(g_free.begin()+i);
      g_used += need;
      return {off, need};
    }
  }
  return {~0ull,0};
}

static void free_slice(Slice s) {
  std::lock_guard<std::mutex> lk(g_mtx);
  g_free.push_back(s);
  coalesce_sorted(g_free);
  g_used -= s.len;
}

int main(int argc, char** argv){
  const char* sock = "/tmp/gpu_pool.sock";
  int device = 0;
  uint64_t pool_bytes = 24ull<<30;
  for (int i=1;i<argc;i++){
    if (!std::strcmp(argv[i],"--endpoint") && i+1<argc) sock=argv[++i];
    else if (!std::strcmp(argv[i],"--device") && i+1<argc) device=std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i],"--pool-bytes") && i+1<argc) pool_bytes = std::stoull(argv[++i]);
  }

  cuInit(0);
  check(cuDeviceGet(&g_dev, device), "cuDeviceGet");
  check(cuCtxCreate(&g_ctx, 0, g_dev), "cuCtxCreate");

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  check(cuMemGetAllocationGranularity(&g_gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity");
  g_pool_bytes = ((pool_bytes + g_gran - 1)/g_gran)*g_gran;

  check(cuMemCreate(&g_pool, g_pool_bytes, &prop, 0), "cuMemCreate");

  int fd = -1;
  check(cuMemExportToShareableHandle(&fd, g_pool, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), "export");
  g_share_fd = fd;

  g_free = { Slice{0, g_pool_bytes} };

  RpcServer srv(sock);
  std::cout<<"gpu_poold listening "<<sock<<" pool="<<g_pool_bytes<<"B gran="<<g_gran<<"\n";

  while (true){
    int cfd = srv.accept_one();
    std::thread([cfd](){
      int cmd=0; if (::read(cfd, &cmd, sizeof(cmd))<=0){ ::close(cfd); return; }
      if (cmd==1){ // HELLO
        int32_t st=0; ::write(cfd, &st, sizeof(st));
        uint64_t meta[2] = { g_gran, g_pool_bytes };
        ::write(cfd, meta, sizeof(meta));
        // send fd via SCM_RIGHTS
        struct msghdr msg{}; struct iovec iov; char ch='F';
        iov.iov_base=&ch; iov.iov_len=1; msg.msg_iov=&iov; msg.msg_iovlen=1;
        char cmsgbuf[CMSG_SPACE(sizeof(int))];
        msg.msg_control=cmsgbuf; msg.msg_controllen=sizeof(cmsgbuf);
        struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level=SOL_SOCKET; cmsg->cmsg_type=SCM_RIGHTS; cmsg->cmsg_len=CMSG_LEN(sizeof(int));
        *(int*)CMSG_DATA(cmsg)=g_share_fd;
        msg.msg_controllen=CMSG_SPACE(sizeof(int));
        sendmsg(cfd, &msg, 0);
      } else if (cmd==2){ // ALLOC
        uint64_t nbytes=0; ::read(cfd, &nbytes, sizeof(nbytes));
        Slice s = alloc_slice(nbytes);
        int32_t st = (s.len==0)?-1:0;
        ::write(cfd, &st, sizeof(st));
        ::write(cfd, &s, sizeof(s));
      } else if (cmd==3){ // FREE
        Slice s{}; ::read(cfd, &s, sizeof(s));
        free_slice(s);
        int32_t st=0; ::write(cfd, &st, sizeof(st));
      } else if (cmd==4){ // STATS
        uint64_t used = g_used.load();
        int32_t st=0; ::write(cfd, &st, sizeof(st));
        ::write(cfd, &used, sizeof(used));
        ::write(cfd, &g_pool_bytes, sizeof(g_pool_bytes));
        ::write(cfd, &g_gran, sizeof(g_gran));
      } else {
        int32_t st=-2; ::write(cfd, &st, sizeof(st));
      }
      ::close(cfd);
    }).detach();
  }
  return 0;
}
