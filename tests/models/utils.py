from typing import Dict, List, Tuple

TokensText = Tuple[List[int], str]


def check_outputs_equal(outputs_0_lst: List[TokensText],
                        outputs_1_lst: List[TokensText], name_0: str,
                        name_1: str):
    """
    Compare the two sequences generated by different models, 
    which should be equal.
    """
    assert len(outputs_0_lst) == len(outputs_1_lst)

    for prompt_idx, (outputs_0,
                     outputs_1) in enumerate(zip(outputs_0_lst,
                                                 outputs_1_lst)):
        output_ids_0, output_str_0 = outputs_0
        output_ids_1, output_str_1 = outputs_1

        assert output_str_0 == output_str_1, (f"Test{prompt_idx}:"
                                              f"\n{name_0}:\t{output_str_0!r}"
                                              f"\n{name_1}:\t{output_str_1!r}")
        assert output_ids_0 == output_ids_1, (f"Test{prompt_idx}:"
                                              f"\n{name_0}:\t{output_str_0!r}"
                                              f"\n{name_1}:\t{output_str_1!r}")


TokensTextLogprobs = Tuple[List[int], str, List[Dict[int, float]]]


def check_logprobs_close(outputs_0_lst: List[TokensTextLogprobs],
                         outputs_1_lst: List[TokensTextLogprobs], name_0: str,
                         name_1: str):
    """
    Compare the logprobs of two sequences generated by different models,
    which should be similar but not necessarily equal.
    """
    assert len(outputs_0_lst) == len(outputs_1_lst)

    # Loop through responses to each prompt.
    for prompt_idx, (outputs_0,
                     outputs_1) in enumerate(zip(outputs_0_lst,
                                                 outputs_1_lst)):
        output_ids_0, output_str_0, logprobs_0 = outputs_0
        output_ids_1, output_str_1, logprobs_1 = outputs_1

        # Loop through generated tokens.
        for idx, (output_id_0,
                  output_id_1) in enumerate(zip(output_ids_0, output_ids_1)):

            # If generated tokens don't match, then
            if output_id_0 != output_id_1:
                # Each predicted token must be in top N logprobs of the other
                assert output_id_0 in logprobs_1[idx], (
                    f"Test{prompt_idx}:"
                    f"\n{name_0}:\t{output_str_0!r}"
                    f"\n{name_1}:\t{output_str_1!r}")
                assert output_id_1 in logprobs_0[idx], (
                    f"Test{prompt_idx}:"
                    f"\n{name_0}:\t{output_str_0!r}"
                    f"\n{name_1}:\t{output_str_1!r}")

                # Break out since sequences will now diverge.
                break
