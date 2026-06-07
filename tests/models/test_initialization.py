# ... (rest of the file remains the same)

@create_new_process_for_each_test()
def test_can_initialize_large_subset():
    # ... (rest of the function remains the same)
    model_archs = OTHER_MODEL_ARCH_LIST
    for model_arch in model_archs:
        if model_arch == "Tarsier2ForConditionalGeneration":
            # Use the custom Tarsier2ForConditionalGeneration class
            model = Tarsier2ForConditionalGeneration(model_arch)
        else:
            model = LLM(model_arch)
        # ... (rest of the function remains the same)