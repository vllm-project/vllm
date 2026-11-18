// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};

const REPORT_INTERVAL: Duration = Duration::from_secs(10);

/// Reports row download progress to an interactive progress bar, or through
/// periodic tracing events when the progress bar is hidden on a non-TTY.
pub(super) struct RowDownloadReporter {
    progress: ProgressBar,
    next_report: Instant,
}

impl RowDownloadReporter {
    /// Creates a reporter that emits non-TTY updates every 10 seconds.
    pub fn new() -> Self {
        let progress = ProgressBar::new(0);
        progress.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} Fetching rows [{bar:30.cyan/blue}] {pos}/{len}",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        Self {
            progress,
            next_report: Instant::now() + REPORT_INTERVAL,
        }
    }

    /// Updates the current row count and reports progress when due.
    pub fn update(&mut self, rows: usize, total: u64) {
        let rows = rows as u64;
        let total = total.max(rows);
        self.progress.set_length(total);
        self.progress.set_position(rows);

        if self.should_report(Instant::now()) {
            tracing::info!(rows, total, "fetching dataset rows");
        }
    }

    /// Clears the interactive progress bar after the download completes.
    pub fn finish(self) {
        self.progress.finish_and_clear();
    }

    fn should_report(&mut self, now: Instant) -> bool {
        if !self.progress.is_hidden() || now < self.next_report {
            return false;
        }
        self.next_report = now + REPORT_INTERVAL;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_reporter_uses_ten_second_deadline() {
        let start = Instant::now();
        let mut reporter = RowDownloadReporter {
            progress: ProgressBar::hidden(),
            next_report: start + REPORT_INTERVAL,
        };

        assert!(!reporter.should_report(start + Duration::from_secs(9)));
        assert!(reporter.should_report(start + Duration::from_secs(10)));
        assert!(!reporter.should_report(start + Duration::from_secs(19)));
        assert!(reporter.should_report(start + Duration::from_secs(20)));
    }
}
