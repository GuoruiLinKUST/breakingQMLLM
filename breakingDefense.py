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
	os.makedirs('./fig', exist_ok=True)
	rows = [['Prompt', 'Response', 'Score', 'Target in Response', 'Best Loss']]
	avgScore = 0
	TargetInResponse = 0
	for idx, (prompt, target) in enumerate(data_tuples):
		img = torch.randint(0, 256, (3, 256, 256), dtype=torch.float32)
		cleanImg = img.clone()
		img.requires_grad = True
		bestImg = img.clone()
		bestLoss = float('inf')
		opt = AdamW([img], lr=1.0, weight_decay=0.0)
		with tqdm.tqdm(range(1000), total=len(range(1000)), desc=f"Prompt{idx}; Target{target} Loss: inf", dynamic_ncols=True) as pbar:
			for i in range(1000):
				img.requires_grad = True
				messagesNoTarget = [
					{
						"role": "user",
						"content": [
							{'type': 'image'},
							{'type': 'text', "text": prompt}
						]
					}
				]
				promptNoTarget = processor.apply_chat_template(
					messagesNoTarget,
					tokenize=False,
					add_generation_prompt=True
				)
				promptWithTarget = promptNoTarget + target
				inputs = process_with_unitok_diff(processor, images=img, text=promptWithTarget, return_tensors="pt").to(model.device).to(model.dtype)
				inputsNoTarget = process_with_unitok_diff(processor, images=img, text=promptNoTarget, return_tensors="pt").to(model.device).to(model.dtype)
				output = model(**inputs, bpda=True)
				labels = inputs.input_ids.clone()
				minLabelIdx = labels.shape[-1]
				for _ in range(labels.shape[0]):
					labels[_:_ + 1, :inputsNoTarget.input_ids[_].shape[0]] = -100
					minLabelIdx = min(minLabelIdx, inputsNoTarget.input_ids[_].shape[0])
				shiftLogits = output.logits[:, minLabelIdx - labels.shape[-1] - 1:-1, :].view(-1, output.logits.shape[-1]).float()
				labels = labels[:, minLabelIdx:].view(-1)
				loss = torch.nn.functional.cross_entropy(shiftLogits, labels, reduction='mean')
				if loss.item() < bestLoss:
					bestLoss = loss.item()
					bestImg = img.clone()
				img.grad = None
				loss.backward()
				assert img.grad is not None
				opt.step()
				# img.data -= torch.sign(img.grad.data)
				# img.data = torch.clamp(img.data, cleanImg.data - 8, cleanImg.data + 8)
				img.data = torch.clamp(img.data, 0, 255)
				pbar.set_description(f"Prompt{idx}; Target: {target}; Loss: {loss.item()}; Best Loss: {bestLoss}")
				pbar.update()

				if 1 / math.exp(loss.item()) >= 0.99:
					break

		pilImg = torchvision.transforms.functional.to_pil_image(bestImg / 255)
		pilImg.save(f'./fig/{idx}.png', format="PNG")
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
		rows.append([prompt, response, score, int(target in response), bestLoss])
	with open('./rec.csv', 'w+', newline='') as f:
		csv.writer(f).writerows(rows)
