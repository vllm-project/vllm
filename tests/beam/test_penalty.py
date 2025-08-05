import os

import torch
from vllm.beam.penalty import PenaltyComputer, MEOW_CLASSI_IDX
import json

class TestPenaltyComputer:

    def test_penalty_meow(self):
        """Test penalty computation using values from the provided JSON example"""
        classi_idx = MEOW_CLASSI_IDX
        penalty_computer = PenaltyComputer(classi_idx)

        json_path = os.path.join(os.path.dirname(__file__), 'examples/penalty_meow.json')
        with open(json_path, 'r') as f:
            prob_data = json.load(f)


        prob_data_cleaned = {}
        for key, value  in prob_data.items():
            key = key.replace('prob_', 'annotations_')
            prob_data_cleaned[key] = value

        num_candidates = 3
        num_classifiers = len(classi_idx)
        prob_GC = torch.zeros(num_candidates, num_classifiers)
        logit_GC = torch.zeros(num_candidates, num_classifiers)
        for classifier_name, idx in classi_idx.items():
            if classifier_name in prob_data_cleaned:
                for i, cand_key in enumerate(prob_data_cleaned[classifier_name]):
                    prob_GC[i, idx] = prob_data_cleaned[classifier_name][cand_key]


        penalties = penalty_computer.compute(logit_GC, prob_GC=prob_GC)
        

        expected_penalties = [0, 2538.0546875, 1083.516845703125]
        
        # Check that penalties match expected values (with some tolerance)
        for i, expected in enumerate(expected_penalties):
            assert abs(penalties[i].item() - expected) < 1.0, f"Candidate {i}: expected {expected}, got {penalties[i].item()}"
    
