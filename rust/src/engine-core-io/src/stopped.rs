// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Whether the thread stopped, set as it exits.
#[derive(Default)]
pub(crate) struct Stopped {
    stopped: Mutex<bool>,
    condvar: Condvar,
}

/// Sets [`Stopped`] when dropped, however the thread exits.
pub(crate) struct StoppedGuard(pub(crate) Arc<Stopped>);

impl Drop for StoppedGuard {
    fn drop(&mut self) {
        *self.0.stopped.lock().unwrap() = true;
        self.0.condvar.notify_all();
    }
}

impl Stopped {
    pub(crate) fn get(&self) -> bool {
        *self.stopped.lock().unwrap()
    }

    /// Wait until stopped, for at most `timeout` if given; returns whether
    /// it stopped.
    pub(crate) fn wait(&self, timeout: Option<Duration>) -> bool {
        let stopped = self.stopped.lock().unwrap();
        let stopped = match timeout {
            Some(timeout) => {
                (self.condvar.wait_timeout_while(stopped, timeout, |s| !*s)).unwrap().0
            }
            None => self.condvar.wait_while(stopped, |s| !*s).unwrap(),
        };
        *stopped
    }
}
