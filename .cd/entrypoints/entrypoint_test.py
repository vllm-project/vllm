# SPDX-License-Identifier: Apache-2.0
import sys

from entrypoints.entrypoint_base import EntrypointBase


class EntrypointTest(EntrypointBase):

    def run(self):
        print("[INFO] Test mode: keeping container active. "
              "Press Ctrl+C to exit.")
        try:
            while True:
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            print("Exiting test mode.")
            sys.exit(0)
