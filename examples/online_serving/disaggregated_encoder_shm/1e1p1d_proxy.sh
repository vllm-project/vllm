#!/bin/bash
python ../disaggregated_encoder/disagg_epd_proxy.py \
      --encode-servers-urls "http://127.0.0.1:23001" \
      --prefill-servers-urls "http://127.0.0.1:33001" \
      --decode-servers-urls "http://127.0.0.1:43001" \
      --host 127.0.0.1 \
      --port 8001
