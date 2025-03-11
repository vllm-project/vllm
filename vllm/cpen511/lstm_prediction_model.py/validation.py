import torch
from config import *

targetpath = path_keeper['targetpath']
def validate_model(network, data_iterator, relevant_keys, computing_device="cpu", initial_state=None, parts=1):
    network.eval()
    with open(targetpath + "/prediction_output.log", "w") as po:
        po.write(f'correct, label, prediction\n')
        
    accuracy_metrics_10 = []
    for i, batch_data in enumerate(data_iterator):
        accuracy_10 = process_batch(i, batch_data, network, computing_device, initial_state, relevant_keys)
        accuracy_metrics_10.append(accuracy_10)
        
        if i >= len(data_iterator) * parts:
            break
        
        # output progress
        if i % 65536 == 0 and i != 0:
            print(f"Validation Iteration {i} of {len(data_iterator)}")

    average_accuracy_10 = torch.tensor(accuracy_metrics_10).mean()

    return average_accuracy_10

def process_batch(batch_index, batch_data, network, device, state, keys):
    # print(f"Processing batch {batch_index}")

    batch_data = [item.to(device) for item in batch_data]
    input_data = batch_data[:-1]
    labels = batch_data[-1]
    labels = torch.argmax(labels, dim = 1)
    
    predictions, state = network.predict(input_data, state)
    accuracy_10 = compute_accuracy(predictions, labels, keys)

    return accuracy_10

def compute_accuracy(predictions, labels, keys):
    # print(predictions)
    # print(labels)
    combined_data = list(zip(labels, predictions))
    # count_correct = sum([1 for label, predicted in combined_data 
                        #  if label.item() in keys and label in predicted])
    count_correct = 0              
    with open(targetpath + "/prediction_output.log", "a") as po:
        for label, prediction in combined_data:
        # print(f"label: {label}, prediction: {prediction}")
            
            # change the prediction from tensor to list
            prediction = prediction.tolist()
            
            # Decode the label and prediction
            try:
                label = label_encoder_deltas.inverse_transform([label.item()])[0]
                prediction = [label_encoder_deltas.inverse_transform([pred])[0] for pred in prediction]
            except:
                print(f"Error: label {label} or prediction {prediction} not in keys")
                continue
            
            if label in prediction:
                count_correct += 1
                po.write(f'y,{label},{prediction}\n')
            else:
                po.write(f'n,{label},{prediction}\n')
            
    return count_correct / len(labels)