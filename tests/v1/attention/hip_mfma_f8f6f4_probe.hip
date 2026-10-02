// Pin the register layout of v_mfma_scale_f32_16x16x128_f8f6f4 on gfx950 for
// FP4 E2M1 A x FP8 E4M3 B with per-32 E8M0 scales on both operands. Every
// piece of ops/mxfp4_mla_native.hip depends on this layout, so it is measured.
//
// Method: fill each lane's registers with known random codes, run one
// instruction, then on the host rebuild the A, B matrices under every
// candidate mapping of (lane, register element) -> (row|col, k) and report
// which candidate reproduces the hardware's D. Step 1 fixes all scales at
// 1.0 (E8M0 127) to find the data layout alone; step 2 randomizes scales and
// tests which lane supplies the scale for each (row|col, 32-wide k block).
//
// A physical element n (0..31) of lane l is nibble n of its 128-bit A operand
// (dword n/8, bits 4*(n%8)); B physical element n is byte n of its 256-bit B
// operand (dword n/4, bits 8*(n%4)). Row/col is assumed l % 16.
//
// Build: hipcc --offload-arch=gfx950 -O2 hip_mfma_f8f6f4_probe.hip -o probe
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int v8i;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float v4f;

constexpr int FMT_E4M3 = 0, FMT_E2M1 = 4;

// Raw registers in, D out. areg: [64][4] dwords, breg: [64][8] dwords.
__global__ void run(const uint32_t* areg, const uint32_t* breg, const int* sa,
                    const int* sb, float* d) {
  const int l = threadIdx.x;
  v8i a = {0, 0, 0, 0, 0, 0, 0, 0}, b;
  for (int i = 0; i < 4; ++i) a[i] = areg[l * 4 + i];
  for (int i = 0; i < 8; ++i) b[i] = breg[l * 8 + i];
  v4f acc = {0, 0, 0, 0};
  acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc, FMT_E2M1, FMT_E4M3,
                                                         0, sa[l], 0, sb[l]);
  for (int i = 0; i < 4; ++i) d[l * 4 + i] = acc[i];
}

static double e2m1(unsigned v) {
  double s = (v & 8) ? -1 : 1;
  int e = (v >> 1) & 3, m = v & 1;
  return s * (e == 0 ? 0.5 * m : (1 + 0.5 * m) * std::ldexp(1.0, e - 1));
}
static double e4m3(unsigned v) {
  double s = (v & 0x80) ? -1 : 1;
  int e = (v >> 3) & 15, m = v & 7;
  return s * (e == 0 ? (m / 8.0) * std::ldexp(1.0, -6) : (1 + m / 8.0) * std::ldexp(1.0, e - 7));
}

// Candidate k for lane-block c (0..3) and physical element n (0..31).
static int kmap(int cand, int c, int n) {
  int nn = (cand & 1) ? (n ^ 1) : n;  // odd candidates swap nibble/byte pairs
  switch (cand >> 1) {
    case 0: return 32 * c + nn;
    case 1: return 16 * c + (nn % 16) + 64 * (nn / 16);
    case 2: return 8 * c + (nn % 8) + 32 * (nn / 8);
    case 3: return 4 * c + (nn % 4) + 16 * (nn / 4);
    case 4: return 2 * c + (nn % 2) + 8 * (nn / 2);
    default: return c + 4 * nn;
  }
}
constexpr int NCAND = 12;
static const char* kname[6] = {"32c+n", "16c+n%16+64(n/16)", "8c+n%8+32(n/8)",
                               "4c+n%4+16(n/4)", "2c+n%2+8(n/2)", "c+4n"};

static uint32_t areg[64 * 4], breg[64 * 8];
static int sa[64], sb[64];
static float dhw[64 * 4];

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
  printf("HIP %s @%d\n", hipGetErrorString(_e), __LINE__); exit(2); } } while (0)

static void launch() {
  uint32_t *da, *db; int *dsa, *dsb; float* dd;
  CK(hipMalloc(&da, sizeof areg)); CK(hipMalloc(&db, sizeof breg));
  CK(hipMalloc(&dsa, sizeof sa)); CK(hipMalloc(&dsb, sizeof sb)); CK(hipMalloc(&dd, sizeof dhw));
  CK(hipMemcpy(da, areg, sizeof areg, hipMemcpyHostToDevice));
  CK(hipMemcpy(db, breg, sizeof breg, hipMemcpyHostToDevice));
  CK(hipMemcpy(dsa, sa, sizeof sa, hipMemcpyHostToDevice));
  CK(hipMemcpy(dsb, sb, sizeof sb, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(run, dim3(1), dim3(64), 0, 0, da, db, dsa, dsb, dd);
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(dhw, dd, sizeof dhw, hipMemcpyDeviceToHost));
  hipFree(da); hipFree(db); hipFree(dsa); hipFree(dsb); hipFree(dd);
}

static unsigned a_el(int l, int n) { return (areg[l * 4 + n / 8] >> (4 * (n % 8))) & 15; }
static unsigned b_el(int l, int n) { return (breg[l * 8 + n / 4] >> (8 * (n % 4))) & 255; }

// Hardware D for (row i, col j) under output map dm: 0 = lane j + 16*(i/4), elem i%4;
// 1 = lane i + 16*(j/4), elem j%4 (transposed).
static double dget(int dm, int i, int j) {
  return dm == 0 ? dhw[(j + 16 * (i / 4)) * 4 + i % 4] : dhw[(i + 16 * (j / 4)) * 4 + j % 4];
}

// Scale for (row|col, kblock b) comes from lane (row|col) + 16*perm[b].
static int pa[4] = {0, 1, 2, 3}, pb[4] = {0, 1, 2, 3};

// Relative error of the reference built under (ca, cb, dm, scale mode sm) vs hardware.
// sm 0: scales ignored (all 1.0). sm 1: scales per pa / pb.
static double score(int ca, int cb, int dm, int sm) {
  static double A[16][128], B[128][16];
  for (int l = 0; l < 64; ++l)
    for (int n = 0; n < 32; ++n) {
      int k = kmap(ca, l / 16, n); A[l % 16][k] = e2m1(a_el(l, n));
      k = kmap(cb, l / 16, n);     B[k][l % 16] = e4m3(b_el(l, n));
    }
  double worst = 0, mag = 0;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      double s = 0;
      for (int k = 0; k < 128; ++k) {
        double w = 1;
        if (sm == 1)
          w = std::ldexp(1.0, sa[i + 16 * pa[k / 32]] - 127) *
              std::ldexp(1.0, sb[j + 16 * pb[k / 32]] - 127);
        s += A[i][k] * B[k][j] * w;
      }
      mag = std::fmax(mag, std::fabs(s));
      worst = std::fmax(worst, std::fabs(dget(dm, i, j) - s));
    }
  return worst / (mag + 1e-30);
}

int main() {
  srand(7);
  for (auto& x : areg) x = (uint32_t)rand() * 2654435761u ^ rand();
  for (int l = 0; l < 64; ++l)
    for (int d = 0; d < 8; ++d) {
      uint32_t w = 0;
      for (int b = 0; b < 4; ++b) {
        unsigned v;
        do { v = rand() & 0xFF; } while ((v & 0x7F) == 0x7F);  // no NaN
        w |= v << (8 * b);
      }
      breg[l * 8 + d] = w;
    }

  // A misplaced element costs ~1e-2 of max|D|; fp32 summation order ~1e-5.
  const double TOL = 1e-3;

  // Step 1: data layout with unit scales. D is invariant under any relabeling
  // of k applied to A and B together, so several candidates may tie; step 2's
  // scales, which attach to absolute 32-wide k blocks, break the tie.
  for (int l = 0; l < 64; ++l) sa[l] = sb[l] = 127;
  launch();
  int ties[64][3], nt = 0;
  for (int ca = 0; ca < NCAND; ++ca)
    for (int cb = 0; cb < NCAND; ++cb)
      for (int dm = 0; dm < 2; ++dm) {
        double v = score(ca, cb, dm, 0);
        if (v < TOL && nt < 64) { ties[nt][0] = ca; ties[nt][1] = cb; ties[nt][2] = dm; ++nt;
          printf("STEP 1 match %.2e  A %s%s | B %s%s | D %s\n", v, kname[ca >> 1],
                 (ca & 1) ? "^1" : "", kname[cb >> 1], (cb & 1) ? "^1" : "",
                 dm == 0 ? "[4c+i][l%16]" : "[l%16][4c+i]"); }
      }
  if (nt == 0) { printf("NO DATA LAYOUT MATCHES\n"); return 1; }

  // Step 2: random scales; search every data candidate x every block->lane perm.
  for (int l = 0; l < 64; ++l) { sa[l] = 118 + rand() % 19; sb[l] = 118 + rand() % 19; }
  launch();
  int perms[24][4], np = 0, p[4] = {0, 1, 2, 3};
  do { for (int i = 0; i < 4; ++i) perms[np][i] = p[i]; ++np; }
  while (std::next_permutation(p, p + 4));
  int found = 0;
  for (int t = 0; t < nt; ++t)
    for (int x = 0; x < 24; ++x)
      for (int y = 0; y < 24; ++y) {
        for (int i = 0; i < 4; ++i) { pa[i] = perms[x][i]; pb[i] = perms[y][i]; }
        double v = score(ties[t][0], ties[t][1], ties[t][2], 1);
        if (v < TOL) {
          ++found;
          printf("STEP 2 match %.2e  A %s%s | B %s%s | scale lanes A {%d%d%d%d} B {%d%d%d%d}\n",
                 v, kname[ties[t][0] >> 1], (ties[t][0] & 1) ? "^1" : "",
                 kname[ties[t][1] >> 1], (ties[t][1] & 1) ? "^1" : "",
                 pa[0], pa[1], pa[2], pa[3], pb[0], pb[1], pb[2], pb[3]);
        }
      }
  printf(found ? "LAYOUT CONFIRMED (%d match)\n" : "NO SCALE MAPPING MATCHES\n", found);
  return found ? 0 : 1;
}
