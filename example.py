from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# MODEL="nm-testing/Meta-Llama-3-8B-Instruct-W4-Group128-A16-Test"
# MODEL="nm-testing/Meta-Llama-3-8B-FP8-compressed-tensors-test-bos"
# MODEL="neuralmagic/Meta-Llama-3-8B-Instruct-FP8"
# MODEL="nm-testing/Meta-Llama-3-8B-Instruct-GPTQ"
MODEL="llm-compressor/Meta-Llama-3-8B-Instruct-W8A8-FP8-BOS"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

PROMPT = "Question: Steve finds 100 gold bars while visiting Oregon. He wants to distribute his gold bars evenly to his 4 friends. If 20 gold bars were lost on the way back to San Diego, how many gold bars will each of his 4 friends get when he returns?\nAnswer: He only has 100 - 20 = <<100-20=80>>80 gold bars after losing 20 of them.\nHe then gives each of his friends 80 ÷ 4 = <<80/4=20>>20 gold bars.\n#### 20\n\nQuestion: In a week, Mortdecai collects 8 dozen  eggs every Tuesday and Thursday, and he delivers 3 dozen  eggs to the market and 5 dozen eggs to the mall. He then uses 4 dozen eggs to make a pie every Saturday. Mortdecai donates the remaining eggs to the charity by Sunday. How many  eggs does he donate to the charity?\nAnswer: Mortdecai collects a total of 8x2 = <<8*2=16>>16 dozens of eggs.\nHe sells a total of 3 + 5 = <<3+5=8>>8 dozens of eggs.\nSo, 16 - 8 = <<16-8=8>>8 dozens of eggs are left.\nAfter using 4 dozens of eggs to make a pie, 8 - 4 = <<8-4=4>>4 dozens of eggs are left.\nSince there are 12 in 1 dozen, then Mortdecai donates 4 x 12 = <<4*12=48>>48 pieces of eggs to the charity.\n#### 48\n\nQuestion: Corey downloaded two movie series from his Netflix account with 12 and 14 seasons per series, respectively. However, in the week, his computer got a mechanical failure, and he lost two episodes from each season for both series. If each season in the movie series that Corey downloaded had 16 episodes, how many episodes remained after the computer's mechanical failure?\nAnswer: In the first movie series with 12 seasons, after the mechanical failure, the number of episodes that Corey lost is 2*12 = <<2*12=24>>24\nOriginally, the movie series with 12 seasons had 12*16 = <<12*16=192>>192 episodes.\nAfter the mechanical failure, Corey had 192-24 = <<192-24=168>>168 episodes remaining in the first movie series.\nSimilarly, the 14 season movie series also had 14*2 = <<14*2=28>>28 lost after the computer's mechanical failure.\nOriginally, the movie series with 14 seasons has 14*16 = <<14*16=224>>224 episodes.\nThe mechanical failure of the computer reduced the number of episodes in the 14 season movie series to 224-28 = <<224-28=196>>196\nAfter the loss, Corey had 196+168 = <<196+168=364>>364 episodes remaining from the two movie series he had downloaded.\n#### 364\n\nQuestion: There were 18 students assigned in a minibus for a field trip. Eight of these students were boys. On the day of the field trip, the number of girls and boys was the same since some of the girls were not able to join the trip. How many girls were not able to join the field trip?\nAnswer: 8 boys + 8 girls = <<8+8=16>>16 students joined the field trip.\nThus, 18 - 16 = <<18-16=2>>2 girls were not able to join the field trip.\n#### 2\n\nQuestion: There are 200 more red apples than green apples in a grocery store. A truck arrives and delivers another 340 green apples. If there were originally 32 green apples, how many more green apples than red apples are there in the store now?\nAnswer: There are 200 + 32 = <<200+32=232>>232 red apples.\nAfter the delivery there are 340 + 32 = <<340+32=372>>372 green apples\nThere are now 372 - 232 = <<372-232=140>>140 more green apples than red apples now.\n#### 140\n\nQuestion: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\nAnswer:"
PARAMS = SamplingParams(max_tokens=100, temperature=0)

TOKENS = tokenizer(PROMPT, add_special_tokens=False).input_ids
TOKENS_WITH_BOS = tokenizer(PROMPT, add_special_tokens=True).input_ids

model = LLM(MODEL)

print("\n\n========================================")
print(tokenizer.decode(TOKENS))
print(TOKENS[:10])
print(TOKENS_WITH_BOS[:10])
print("========================================\n\n")

output = model.generate(prompt_token_ids=[TOKENS], sampling_params=PARAMS)
print(f"===== TOKENS: {output[0].outputs[0].text}")

output = model.generate(prompt_token_ids=[TOKENS_WITH_BOS], sampling_params=PARAMS)
print(f"===== TOKENS WITHOUT WITH BOS: {output[0].outputs[0].text}")
