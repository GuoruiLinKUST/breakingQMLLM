import csv
import math
import os

import PIL.Image
import torch
import torchvision.transforms.functional
import tqdm
from datasets import disable_caching, load_dataset
from transformers import LlavaProcessor
from torch.optim import AdamW

from unitok_qllava import LlavaForConditionalGeneration
from utils.process import process_with_unitok, process_with_unitok_diff
from strong_reject import evaluate

if __name__ == '__main__':
	disable_caching()
	dataset = load_dataset("walledai/AdvBench", token=os.getenv('HF_TOKEN', default=None))
	data_tuples = []
	for example in dataset['train']:
		data_tuples.append((example['prompt'], example['target']))
	model_path = "vincentchao/qmllm_unitok"
	model = LlavaForConditionalGeneration.from_pretrained(
		model_path,
		torch_dtype=torch.float16,
	).cuda()
	processor = LlavaProcessor.from_pretrained(
		model_path,
	)
	model.eval()
	model.requires_grad_(False)
	rows = [['Prompt', 'Response', 'Score', 'Target in Response']]
	avgScore = 0
	TargetInResponse = 0
	for idx, (prompt, target) in enumerate(data_tuples):
		if os.path.exists(f'./fig/{idx}.png'):
			pilImg = PIL.Image.open(f'./fig/{idx}.png')
			conversation = [
				{
					"role": "user",
					"content": [
						{"type": "text", "text": prompt},
						{"type": "image"},
					],
				},
			]
			inputs = process_with_unitok(processor,
										 images=pilImg,
										 text=processor.apply_chat_template(conversation, add_generation_prompt=True),
										 return_tensors="pt").to(model.device).to(model.dtype).to(model.dtype)

			# Generate response
			input_length = inputs['input_ids'].shape[1]

			with torch.no_grad():
				output = model.generate(**inputs, max_new_tokens=512)

			response = processor.decode(output[0][input_length:], skip_special_tokens=True)
			with torch.no_grad():
				score = evaluate.strongreject_finetuned({"forbidden_prompt": [prompt], "response": [response]})['score'][0]
				avgScore += score
				if target in response:
					TargetInResponse += 1
			print(f'Prompt: {prompt}\nReponse: {response}\nScore: {score}; Target in Response: {target in response}; Current Avg. Score: {avgScore / (idx + 1)}; Current Avg. TIR: {TargetInResponse / (idx + 1)}')
			rows.append([prompt, response, score, int(target in response)])
	with open('./recEval.csv', 'w+', newline='') as f:
		csv.writer(f).writerows(rows)
