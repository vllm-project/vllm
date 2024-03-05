from vllm import LLM, SamplingParams
import argparse

def main():

    parser = argparse.ArgumentParser(description='Example script with command-line arguments')

    parser.add_argument('--model', '-m', type=int, default=0, help='Model ID [0-2]')
    parser.add_argument('--tensor_parallel_size', '-t', type=int, default=1, help='tensor parallel size')

    args = parser.parse_args()

    # These are the verified working models.
    models = ["BlackSamorez/TinyLlama-1_1B-Chat-v1_0-AQLM-2Bit-1x16-hf", "BlackSamorez/Llama-2-7b-AQLM-2Bit-1x16-hf", "BlackSamorez/Llama-2-7b-AQLM-2Bit-2x8-hf"]

    model = LLM(models[args.model], enforce_eager=True, tensor_parallel_size=args.tensor_parallel_size)
 
    # this has the dimensions 0 and 1 transposed for the codes, and we don't currently support 8x8 anyway.
    #model = LLM("BlackSamorez/Llama-2-7b-AQLM-2Bit-8x8-hf", enforce_eager=True)
    # this model hangs, need to investigate.
    #model = LLM("BlackSamorez/Mixtral-8x7B-Instruct-v0_1-AQLM-2Bit-1x16-hf", enforce_eager=True)

    # These have custom code and no quantization_config block.
    #model = LLM("BlackSamorez/Llama-2-13b-AQLM-2Bit-1x16-hf", enforce_eager=True, trust_remote_code=True)
    #model = LLM("BlackSamorez/Mixtral-8x7b-AQLM-2Bit-1x16-hf", enforce_eager=True)

    sampling_params = SamplingParams(max_tokens=100, temperature=0)
    outputs = model.generate("Hello my name is", sampling_params=sampling_params)
    print(outputs[0].outputs[0].text)

if __name__ == '__main__':
    main()