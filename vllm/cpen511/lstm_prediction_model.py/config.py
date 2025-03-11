config = {
    "batch_size": 8,
    "sequence_length": 64,
    "target_length": 10,
    "num_output_next": 10
}

from sklearn.preprocessing import LabelEncoder
label_encoder_pc = LabelEncoder()
label_encoder_deltas = LabelEncoder()

hparams = {
    "topPredNum": min(config['num_output_next'], 10),
    # "topPredNum": 1,
    "embed_dim": 128,
    "hidden_dim": 128,
    "output_dim": config['num_output_next'],
    "num_layers": 2,
    "dropout": 0.1, # Was 0.1 before
    "learning_rate": 0.001, # Was 0.001 before
    # "epochs": 40
    "epochs": 10
}

path_keeper = {
    'targetpath': None
}
