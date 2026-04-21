# eager 200
REQUEST_RATES=16 NUM_PROMPTS=200 INPUT_LEN=400 OUTPUT_LEN=150 bash /workspace/hero/hero_EPD/test/epd_performance/epd_test_num.sh --epd --eager --encoders 4 --pds 4; wait 5;
bash /workspace/hero/hero_EPD/test/epd_performance/epd_test_num.sh --cleanup; bash /workspace/hero/hero_EPD/test/clean.sh; wait 3;

REQUEST_RATES=16 NUM_PROMPTS=200 INPUT_LEN=400 OUTPUT_LEN=150 bash /workspace/hero/hero_EPD/test/epd_performance/epd_test_num.sh --baseline --eager --encoders 4 --pds 4; wait 5;
bash /workspace/hero/hero_EPD/test/epd_performance/epd_test_num.sh --cleanup; bash /workspace/hero/hero_EPD/test/clean.sh; wait 3;

REQUEST_RATES=20 NUM_PROMPTS=200 INPUT_LEN=400 OUTPUT_LEN=150 bash /workspace/hero/hero_EPD/test/epd_performance/epd_test_num.sh --epd --eager --encoders 4 --pds 4; wait 5;
bash /workspace/hero/hero_EPD/test/epd_performance/epd_test_num.sh --cleanup; bash /workspace/hero/hero_EPD/test/clean.sh; wait 3;

REQUEST_RATES=20 NUM_PROMPTS=200 INPUT_LEN=400 OUTPUT_LEN=150 bash /workspace/hero/hero_EPD/test/epd_performance/epd_test_num.sh --baseline --eager --encoders 4 --pds 4; wait 5;
bash /workspace/hero/hero_EPD/test/epd_performance/epd_test_num.sh --cleanup; bash /workspace/hero/hero_EPD/test/clean.sh; wait 3;
