import torch
from vllm.beam.debug import BeamDebugInfo
from vllm.beam.utils import filter_missing_classis

classi_names = ['annotations_adult_content', 'annotations_adult_content_v2', 'annotations_bad', 'annotations_bad_memory', 'annotations_bad_response', 'annotations_boring', 'annotations_broken_link', 'annotations_chosen_response', 'annotations_conspiracy_theories', 'annotations_contains_factual_information', 'annotations_contains_factual_information_that_may_change_with_time', 'annotations_custom_feedback', 'annotations_depressing', 'annotations_dislike', 'annotations_disrespectful_towards_anyone', 'annotations_disrespectful_towards_sensitive_groups', 'annotations_disturbing', 'annotations_disturbing_v2', 'annotations_diverting_communication', 'annotations_does_not_follow_instruction', 'annotations_doesnt_drive_conversation_forward', 'annotations_eatingdisorder', 'annotations_empathetic', 'annotations_ends_chat_early', 'annotations_engaging', 'annotations_especially_in_character', 'annotations_five_star', 'annotations_flag', 'annotations_follows_instruction_correctly', 'annotations_follows_instruction_incorrectly', 'annotations_four_star', 'annotations_funny', 'annotations_good!', 'annotations_good_response', 'annotations_great_response', 'annotations_harmful_promotes_hatespeech_red', 'annotations_harmful_promotes_physical_harm_to_others_red', 'annotations_harmful_promotes_selfharm', 'annotations_harmful_promotes_selfharm_red', 'annotations_harmful_promotes_terrorism', 'annotations_harmful_promotes_terrorism_red', 'annotations_helpful', 'annotations_i_dislike_this_image', 'annotations_i_hate_this_image', 'annotations_i_like_this_image', 'annotations_i_love_this_image', 'annotations_image_contains_text_that_is_unreadable_or_in_unknown_language', 'annotations_image_five_star', 'annotations_image_four_star', 'annotations_image_has_noticeable_defects', 'annotations_image_is_hard_to_understand', 'annotations_image_is_missing_key_elements_described_in_prompt', 'annotations_image_is_not_pleasing_to_the_eye', 'annotations_image_is_photorealistic', 'annotations_image_is_poorly_cropped', 'annotations_image_is_very_well_drawn_painted_photographed', 'annotations_image_may_be_disturbing_to_some_people', 'annotations_image_may_be_inappropriate_to_some_people', 'annotations_image_one_star', 'annotations_image_three_star', 'annotations_image_two_star', 'annotations_immoral', 'annotations_inaccurate', 'annotations_initiates_ending_chat', 'annotations_initiates_talking_about_adult_content', 'annotations_interesting', 'annotations_leak', 'annotations_like', 'annotations_long', 'annotations_looping', 'annotations_misleading', 'annotations_missing_factual_information', 'annotations_no_instruction_given', 'annotations_nonsense', 'annotations_nonsense_gd', 'annotations_ok_response', 'annotations_one_star', 'annotations_out_of_character', 'annotations_out_of_character_gd', 'annotations_pedophilia', 'annotations_phantom_context', 'annotations_politically_contentious', 'annotations_porn', 'annotations_potentially_controversial', 'annotations_potentially_harmful', 'annotations_potentially_harmful_financial_advice', 'annotations_potentially_harmful_medical_advice', 'annotations_potentially_harmful_v2', 'annotations_potentially_misleading', 'annotations_privacy_sensitive', 'annotations_profane', 'annotations_promising_to_do_something_later', 'annotations_racist', 'annotations_repetitive', 'annotations_rude_uncaring', 'annotations_scim', 'annotations_seeks_factual_information', 'annotations_selfharm', 'annotations_sexually_suggestive', 'annotations_sexually_suggestive_M_rated', 'annotations_sexually_suggestive_R_rated', 'annotations_sexually_suggestive_X_rated', 'annotations_sexually_suggestive_v2', 'annotations_short', 'annotations_superlike', 'annotations_swipe_selected', 'annotations_sx2_plus', 'annotations_sx3_plus', 'annotations_sx4_plus', 'annotations_terrible_response', 'annotations_three_star', 'annotations_truncated', 'annotations_two_star', 'annotations_ugly', 'annotations_unhelpful', 'annotations_unhelpful_factual_information', 'annotations_unsafe', 'annotations_violence', 'annotations_wrong_broken_link', 'annotations_wrong_facts', 'annotations_wrong_link', 'annotations_👍', 'annotations_👎', 'annotations_😀', 'annotations_😍', 'annotations_😒', 'annotations_😞', 'annotations_😡', 'annotations_😮', 'annotations_🤗', 'annotations_🤣', 'annotations_🤮', 'chosen_after_swipe_crowd_preference', 'chosen_after_swipe_preference', 'repetitive']

MEOW_CLASSI = [
  {
    "index": 0,
    "name": "repetitive"
  },
  {
    "index": 1,
    "name": "annotations_adult_content"
  },
  {
    "index": 2,
    "name": "annotations_adult_content_v2"
  },
  {
    "index": 3,
    "name": "annotations_bad"
  },
  {
    "index": 4,
    "name": "annotations_bad_memory"
  },
  {
    "index": 5,
    "name": "annotations_bad_response"
  },
  {
    "index": 6,
    "name": "annotations_boring"
  },
  {
    "index": 7,
    "name": "annotations_broken_link"
  },
  {
    "index": 8,
    "name": "annotations_chosen_response"
  },
  {
    "index": 9,
    "name": "annotations_conspiracy_theories"
  },
  {
    "index": 10,
    "name": "annotations_contains_factual_information"
  },
  {
    "index": 11,
    "name": "annotations_contains_factual_information_that_may_change_with_time"
  },
  {
    "index": 12,
    "name": "annotations_custom_feedback"
  },
  {
    "index": 13,
    "name": "annotations_depressing"
  },
  {
    "index": 14,
    "name": "annotations_dislike"
  },
  {
    "index": 15,
    "name": "annotations_disrespectful_towards_anyone"
  },
  {
    "index": 16,
    "name": "annotations_disrespectful_towards_sensitive_groups"
  },
  {
    "index": 17,
    "name": "annotations_disturbing"
  },
  {
    "index": 18,
    "name": "annotations_disturbing_v2"
  },
  {
    "index": 19,
    "name": "annotations_diverting_communication"
  },
  {
    "index": 20,
    "name": "annotations_does_not_follow_instruction"
  },
  {
    "index": 21,
    "name": "annotations_doesnt_drive_conversation_forward"
  },
  {
    "index": 22,
    "name": "annotations_empathetic"
  },
  {
    "index": 23,
    "name": "annotations_ends_chat_early"
  },
  {
    "index": 24,
    "name": "annotations_engaging"
  },
  {
    "index": 25,
    "name": "annotations_especially_in_character"
  },
  {
    "index": 26,
    "name": "annotations_five_star"
  },
  {
    "index": 27,
    "name": "annotations_flag"
  },
  {
    "index": 28,
    "name": "annotations_follows_instruction_correctly"
  },
  {
    "index": 29,
    "name": "annotations_follows_instruction_incorrectly"
  },
  {
    "index": 30,
    "name": "annotations_four_star"
  },
  {
    "index": 31,
    "name": "annotations_funny"
  },
  {
    "index": 32,
    "name": "annotations_good!"
  },
  {
    "index": 33,
    "name": "annotations_good_response"
  },
  {
    "index": 34,
    "name": "annotations_great_response"
  },
  {
    "index": 35,
    "name": "annotations_harmful_promotes_hatespeech_red"
  },
  {
    "index": 36,
    "name": "annotations_harmful_promotes_physical_harm_to_others_red"
  },
  {
    "index": 37,
    "name": "annotations_harmful_promotes_selfharm"
  },
  {
    "index": 38,
    "name": "annotations_harmful_promotes_selfharm_red"
  },
  {
    "index": 39,
    "name": "annotations_harmful_promotes_terrorism"
  },
  {
    "index": 40,
    "name": "annotations_harmful_promotes_terrorism_red"
  },
  {
    "index": 41,
    "name": "annotations_helpful"
  },
  {
    "index": 42,
    "name": "annotations_i_dislike_this_image"
  },
  {
    "index": 43,
    "name": "annotations_i_hate_this_image"
  },
  {
    "index": 44,
    "name": "annotations_i_like_this_image"
  },
  {
    "index": 45,
    "name": "annotations_i_love_this_image"
  },
  {
    "index": 46,
    "name": "annotations_image_contains_text_that_is_unreadable_or_in_unknown_language"
  },
  {
    "index": 47,
    "name": "annotations_image_five_star"
  },
  {
    "index": 48,
    "name": "annotations_image_four_star"
  },
  {
    "index": 49,
    "name": "annotations_image_has_noticeable_defects"
  },
  {
    "index": 50,
    "name": "annotations_image_is_hard_to_understand"
  },
  {
    "index": 51,
    "name": "annotations_image_is_missing_key_elements_described_in_prompt"
  },
  {
    "index": 52,
    "name": "annotations_image_is_not_pleasing_to_the_eye"
  },
  {
    "index": 53,
    "name": "annotations_image_is_photorealistic"
  },
  {
    "index": 54,
    "name": "annotations_image_is_poorly_cropped"
  },
  {
    "index": 55,
    "name": "annotations_image_is_very_well_drawn_painted_photographed"
  },
  {
    "index": 56,
    "name": "annotations_image_may_be_disturbing_to_some_people"
  },
  {
    "index": 57,
    "name": "annotations_image_may_be_inappropriate_to_some_people"
  },
  {
    "index": 58,
    "name": "annotations_image_one_star"
  },
  {
    "index": 59,
    "name": "annotations_image_three_star"
  },
  {
    "index": 60,
    "name": "annotations_image_two_star"
  },
  {
    "index": 61,
    "name": "annotations_immoral"
  },
  {
    "index": 62,
    "name": "annotations_inaccurate"
  },
  {
    "index": 63,
    "name": "annotations_initiates_ending_chat"
  },
  {
    "index": 64,
    "name": "annotations_initiates_talking_about_adult_content"
  },
  {
    "index": 65,
    "name": "annotations_interesting"
  },
  {
    "index": 66,
    "name": "annotations_leak"
  },
  {
    "index": 67,
    "name": "annotations_like"
  },
  {
    "index": 68,
    "name": "annotations_long"
  },
  {
    "index": 69,
    "name": "annotations_looping"
  },
  {
    "index": 70,
    "name": "annotations_misleading"
  },
  {
    "index": 71,
    "name": "annotations_missing_factual_information"
  },
  {
    "index": 72,
    "name": "annotations_no_instruction_given"
  },
  {
    "index": 73,
    "name": "annotations_nonsense"
  },
  {
    "index": 74,
    "name": "annotations_nonsense_gd"
  },
  {
    "index": 75,
    "name": "annotations_ok_response"
  },
  {
    "index": 76,
    "name": "annotations_one_star"
  },
  {
    "index": 77,
    "name": "annotations_out_of_character"
  },
  {
    "index": 78,
    "name": "annotations_out_of_character_gd"
  },
  {
    "index": 79,
    "name": "annotations_pedophilia"
  },
  {
    "index": 80,
    "name": "annotations_phantom_context"
  },
  {
    "index": 81,
    "name": "annotations_politically_contentious"
  },
  {
    "index": 82,
    "name": "annotations_porn"
  },
  {
    "index": 83,
    "name": "annotations_potentially_controversial"
  },
  {
    "index": 84,
    "name": "annotations_potentially_harmful"
  },
  {
    "index": 85,
    "name": "annotations_potentially_harmful_financial_advice"
  },
  {
    "index": 86,
    "name": "annotations_potentially_harmful_medical_advice"
  },
  {
    "index": 87,
    "name": "annotations_potentially_harmful_v2"
  },
  {
    "index": 88,
    "name": "annotations_potentially_misleading"
  },
  {
    "index": 89,
    "name": "annotations_privacy_sensitive"
  },
  {
    "index": 90,
    "name": "annotations_profane"
  },
  {
    "index": 91,
    "name": "annotations_promising_to_do_something_later"
  },
  {
    "index": 92,
    "name": "annotations_racist"
  },
  {
    "index": 93,
    "name": "annotations_repetitive"
  },
  {
    "index": 94,
    "name": "annotations_rude_uncaring"
  },
  {
    "index": 95,
    "name": "annotations_seeks_factual_information"
  },
  {
    "index": 96,
    "name": "annotations_sexually_suggestive"
  },
  {
    "index": 97,
    "name": "annotations_sexually_suggestive_M_rated"
  },
  {
    "index": 98,
    "name": "annotations_sexually_suggestive_R_rated"
  },
  {
    "index": 99,
    "name": "annotations_sexually_suggestive_X_rated"
  },
  {
    "index": 100,
    "name": "annotations_sexually_suggestive_v2"
  },
  {
    "index": 101,
    "name": "annotations_short"
  },
  {
    "index": 102,
    "name": "annotations_superlike"
  },
  {
    "index": 103,
    "name": "annotations_swipe_selected"
  },
  {
    "index": 104,
    "name": "annotations_terrible_response"
  },
  {
    "index": 105,
    "name": "annotations_three_star"
  },
  {
    "index": 106,
    "name": "annotations_truncated"
  },
  {
    "index": 107,
    "name": "annotations_two_star"
  },
  {
    "index": 108,
    "name": "annotations_ugly"
  },
  {
    "index": 109,
    "name": "annotations_unhelpful"
  },
  {
    "index": 110,
    "name": "annotations_unhelpful_factual_information"
  },
  {
    "index": 111,
    "name": "annotations_unsafe"
  },
  {
    "index": 112,
    "name": "annotations_violence"
  },
  {
    "index": 113,
    "name": "annotations_wrong_broken_link"
  },
  {
    "index": 114,
    "name": "annotations_wrong_facts"
  },
  {
    "index": 115,
    "name": "annotations_wrong_link"
  },
  {
    "index": 116,
    "name": "annotations_\ud83d\udc4d"
  },
  {
    "index": 117,
    "name": "annotations_\ud83d\udc4e"
  },
  {
    "index": 118,
    "name": "annotations_\ud83d\ude00"
  },
  {
    "index": 119,
    "name": "annotations_\ud83d\ude0d"
  },
  {
    "index": 120,
    "name": "annotations_\ud83d\ude12"
  },
  {
    "index": 121,
    "name": "annotations_\ud83d\ude1e"
  },
  {
    "index": 122,
    "name": "annotations_\ud83d\ude21"
  },
  {
    "index": 123,
    "name": "annotations_\ud83d\ude2e"
  },
  {
    "index": 124,
    "name": "annotations_\ud83e\udd17"
  },
  {
    "index": 125,
    "name": "annotations_\ud83e\udd23"
  },
  {
    "index": 126,
    "name": "annotations_\ud83e\udd2e"
  },
  {
    "index": 127,
    "name": "chosen_after_swipe_crowd_preference"
  },
  {
    "index": 128,
    "name": "chosen_after_swipe_preference"
  },
  {
    "index": 129,
    "name": "annotations_scim"
  },
  {
    "index": 130,
    "name": "annotations_sx2_plus"
  },
  {
    "index": 131,
    "name": "annotations_sx3_plus"
  },
  {
    "index": 132,
    "name": "annotations_sx4_plus"
  },
  {
    "index": 133,
    "name": "annotations_selfharm"
  },
  {
    "index": 134,
    "name": "annotations_eatingdisorder"
  }
]
MEOW_CLASSI_IDX = {n["name"]: n["index"] for n in MEOW_CLASSI}

MODEL_SERVER_RANKING = [
        {"name": "chosen_after_swipe_crowd_preference", "pow": 0.5, "negation": False},
        {"name": "chosen_after_swipe_preference", "pow": 0.5, "negation": False},
    ]

def format_filter(filter_params_list):
    keys = ["name", "threshold"]
    if any([len(p) != len(keys) for p in filter_params_list]):
        raise ValueError(f"Expect 2 keys, seeing {filter_params_list=}")
    return [dict(zip(keys, vals)) for vals in filter_params_list]

def format_penalty(penalty_params_list):
    keys = [
        "name",
        "penalty",
        "final_threshold",
        "per_unit_penalty",
    ]
    if any([len(p) != len(keys) for p in penalty_params_list]):
        raise ValueError(f"Expect {len(keys)} keys, seeing {penalty_params_list=}")
    return [dict(zip(keys, vals)) for vals in penalty_params_list]



MODEL_SERVER_PENALTY = format_penalty([
    ("annotations_sexually_suggestive", 15000, 0.1944, 0),
    ("annotations_porn", 15000, 0.0904, 0), 
    ("annotations_porn", 20000, 0.1603, 1),
    ("annotations_racist", 20000, 0.1227, 1),
    ("annotations_disturbing", 15000, 0.1176, 1),
    ("annotations_harmful_promotes_selfharm", 10000, 0.0629, 1),
    ("annotations_harmful_promotes_selfharm", 10001, 0.0268, 1),
    ("annotations_harmful_promotes_terrorism", 10000, 0.0576, 1),
    ("repetitive", 1000, 0.3709, 1),
    ("annotations_nonsense_gd", 1000, 0.3775, 1),
    ("annotations_looping", 1000, 0.0619, 1),
    ("annotations_out_of_character_gd", 1000, 0.3346, 1),
])

DEFAULT_CHAR_SERVER_FILTER = format_filter(
        [
            ("annotations_porn", 0.1024),
            ("annotations_racist", 0.2856),
            ("annotations_disturbing", 0.1851),
            ("annotations_harmful_promotes_selfharm", 0.0629),
            ("annotations_harmful_promotes_terrorism", 0.1009),
        ]
    )




class PenaltyComputer:
    def __init__(self, classi_idx):
        self.classi_idx = classi_idx

        penalty_params = MODEL_SERVER_PENALTY

        # Filter the params that are not found.
        penalty_params = filter_missing_classis(penalty_params, classi_idx, warn=True)
        self.penalty_params = penalty_params
        pnames = [p["name"] for p in penalty_params]
        self.classi_indices = [self.classi_idx[p["name"]] for p in penalty_params]

        if not self.classi_indices:
            print(f"No penalty classifiers {pnames} found. Candidates will not be penalized.")

        self.dtype = torch.float32

        def _tensor(k):
            data = [p[k] for p in penalty_params]
            return torch.tensor(data, dtype=self.dtype, device="cpu")

        self.penalties_P = _tensor("penalty")
        self.thresholds_P = _tensor("final_threshold")
        self.per_unit_penalties_P = _tensor("per_unit_penalty")

    def compute(self, logit_GC, debug_infos_G: list[BeamDebugInfo] = None):
        if not self.classi_indices:
            return torch.zeros_like(logit_GC[:, 0])

        logit_GC = logit_GC[:, self.classi_indices]
        prob_GC = torch.sigmoid(logit_GC)

        over_threshold = (prob_GC > self.thresholds_P).to(self.dtype)
        classifiers_that_are_over_threshold = [
            [self.penalty_params[i]["name"] for i, flag in enumerate(row) if flag]
            for row in over_threshold.bool().tolist()
        ]

        if debug_infos_G is not None:
            for i in range(len(logit_GC )):
                debug_infos_G[i].penalty_classifiers_that_are_over_threshold = classifiers_that_are_over_threshold[i]

        penalty_GC = over_threshold * self.penalties_P * (
        1 + (prob_GC - self.thresholds_P) * self.per_unit_penalties_P
    )

        return penalty_GC.sum(dim=-1)
