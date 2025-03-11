import pandas as pd

def load_traces(data_path, nrows=None, shuffle=False):
    traces = pd.read_csv(data_path, nrows=nrows)

    # change the value from string to int
    print("Converting the string values read from file to integers based 16")
    # traces['pc'] = traces['pc'].apply(lambda x: int(x, 16))
    traces['delta_in'] = traces['delta_in'].apply(lambda x: int(x))
    traces['delta_out'] = traces['delta_out'].apply(lambda x: int(x))

    if shuffle:
        traces = traces.sample(frac=1)

    # Split the data into training and testing sets
    dataset_length = len(traces)
    
    # 75% training, 25% testing
    print("First 75% of the data is used for training, the rest is used for testing")
    train_data = traces[:3*int(dataset_length/4)]
    test_data = traces[3*int(dataset_length/4):]
    
    print("Training data length: ", len(train_data))

    return train_data, test_data