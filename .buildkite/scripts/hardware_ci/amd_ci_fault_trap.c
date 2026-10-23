// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Native fault handler for AMD CI pytest processes, loaded by amd_ci_teardown.py and
// built into the CI image (docker/Dockerfile.rocm).
//
// On SIGSEGV, SIGBUS or SIGABRT it prints the native backtrace, which
// names the library that faulted. Before it is armed, the fault then proceeds as
// it would have (Python's faulthandler, then the default action), so a crash
// during a test still fails the test. Once armed with pytest's exit status, a
// fault is a teardown fault: it exits with that status instead.
//
// Not covered: a stack overflow in a thread other than the one that installed the
// handler (no alternate signal stack there), as with Python's faulthandler.

#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

static const int signals[] = {SIGSEGV, SIGBUS, SIGABRT};
static const char *const names[] = {"SIGSEGV", "SIGBUS", "SIGABRT"};
#define NSIG_TRAPPED (sizeof signals / sizeof signals[0])

static struct sigaction previous[NSIG_TRAPPED];
static volatile sig_atomic_t armed, status, entered;
static int out = STDERR_FILENO;
static char altstack[64 * 1024];

static void say(const char *s) {
  ssize_t unused = write(out, s, strlen(s));
  (void)unused;
}

static void say_int(long v) {
  char buf[24], *p = buf + sizeof buf;
  int neg = v < 0;
  unsigned long u = neg ? -(unsigned long)v : (unsigned long)v;
  *--p = 0;
  do *--p = (char)('0' + u % 10); while (u /= 10);
  if (neg) *--p = '-';
  say(p);
}

static void handle(int sig, siginfo_t *info, void *ctx) {
  size_t i;
  if (entered++) {  // faulted while reporting: finish without the report
    if (armed) _exit(status);
    signal(sig, SIG_DFL);
    raise(sig);
    return;
  }
  say("\n[amd-ci] native fault: ");
  for (i = 0; i < NSIG_TRAPPED; i++)
    if (signals[i] == sig) say(names[i]);
  say(" in pid ");
  say_int(getpid());
  say(armed ? ", after pytest finished\n" : "\n");
  void *frames[64];
  backtrace_symbols_fd(frames, backtrace(frames, 64), out);
  if (armed) {
    say("[amd-ci] a teardown fault, not a test result: exiting with pytest's status ");
    say_int(status);
    say("\n");
    _exit(status);
  }
  entered = 0;
  for (i = 0; i < NSIG_TRAPPED; i++) {
    if (signals[i] != sig) continue;
    sigaction(sig, &previous[i], NULL);
    if (previous[i].sa_flags & SA_SIGINFO) {
      if (previous[i].sa_sigaction) {
        previous[i].sa_sigaction(sig, info, ctx);
        return;
      }
    } else if (previous[i].sa_handler != SIG_DFL && previous[i].sa_handler != SIG_IGN) {
      previous[i].sa_handler(sig);
      return;
    }
  }
  raise(sig);  // a hardware fault would also re-fault on return; raise covers abort()
}

// (Re)install the handler, reporting to fd. Python's faulthandler restores the handler
// it replaced when it is disabled, so the plugin installs again once faulthandler is done.
int amd_ci_trap_install(int fd) {
  stack_t ss = {.ss_sp = altstack, .ss_size = sizeof altstack};
  struct sigaction sa, old;
  size_t i;
  void *warm[1];
  out = fd;
  backtrace(warm, 1);  // loads libgcc_s now, not inside the handler
  sigaltstack(&ss, NULL);
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = handle;
  // SA_NODEFER: a fault while reporting (an unwinder on a broken stack) re-enters the
  // handler, which then exits, instead of the kernel killing the process with it blocked.
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  for (i = 0; i < NSIG_TRAPPED; i++) {
    if (sigaction(signals[i], &sa, &old)) return -1;
    if (old.sa_sigaction != handle) previous[i] = old;
  }
  return 0;
}

void amd_ci_trap_arm(int pytest_status) {
  status = pytest_status;
  armed = 1;
}

void amd_ci_trap_disarm(void) { armed = 0; }
