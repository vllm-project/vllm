# data in gs://woosuk-exp/exp_sec_6
python plot_sec_6_2.py exp_sec_6  --subset n1-sharegpt --duration 3600
python plot_sec_6_2.py exp_sec_6  --subset n1-alpaca --duration 3600
python plot_sec_6_3.py exp_sec_6 --subset parallel --duration 3600
python plot_sec_6_3.py exp_sec_6 --subset beam --duration 3600
python plot_sec_6_4.py exp_sec_6
python plot_sec_6_5.py exp_sec_6

