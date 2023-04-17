# alpaca-13b:
	python plot_normalized_latency.py paper/exp0/exp/alpaca/opt-13b-tp1/n1 --duration 3600 --ylim 1 --format pdf
# alpaca-66b:
	python plot_normalized_latency.py paper/exp4/exp/alpaca/opt-66b-tp4/n1 --duration 3600 --ylim 1 --format pdf
# alpaca-175b:
	python plot_normalized_latency.py paper/exp-175b/exp/alpaca/opt-175b-tp8/n1/ --duration 3600 --ylim 1 --format pdf

# sharegpt-13b:
	python plot_normalized_latency.py paper/exp0/exp/sharegpt/opt-13b-tp1/n1/ --duration 3600 --ylim 1 --format pdf
# sharegpt-66b:
	python plot_normalized_latency.py paper/exp2/exp/sharegpt/opt-66b-tp4/n1 --duration 3600 --ylim 1 --format pdf
# sharegpt-175b:
	python plot_normalized_latency.py paper/exp-175b/exp/sharegpt/opt-175b-tp8/n1 --duration 3600 --ylim 1 --format pdf


# alpaca-n2-13b:
	python plot_normalized_latency.py paper/exp3/exp/alpaca/opt-13b-tp1/n2 --duration 3600 --ylim 1 --format pdf
# alpaca-n4-13b:
	python plot_normalized_latency.py paper/exp3/exp/alpaca/opt-13b-tp1/n4 --duration 3600 --ylim 1 --format pdf
# alpaca-n6-13b:
    python plot_normalized_latency.py paper/exp0/exp/alpaca/opt-13b-tp1/n6 --duration 3600 --ylim 1 --format pdf
	

# alpaca-n2-beam-13b:
	python plot_normalized_latency.py paper/exp1/exp/alpaca/opt-13b-tp1/n2-beam --duration 3600 --ylim 1 --format pdf
# alpaca-n4-beam-13b:
	python plot_normalized_latency.py paper/exp1/exp/alpaca/opt-13b-tp1/n4-beam --duration 3600 --ylim 1 --format pdf
# alapca-n6-beam-13b:
	python plot_normalized_latency.py paper/exp1/exp/alpaca/opt-13b-tp1/n6-beam --duration 3600 --ylim 1 --format pdf
