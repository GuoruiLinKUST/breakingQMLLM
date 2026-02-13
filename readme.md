# Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security

<div align="center">

[[Paper](https://arxiv.org/abs/2511.16229)](https://www.ndss-symposium.org/)

</div>

## 📖 Overview

Q-MLLM is a novel architecture that integrates **two-level vector quantization** to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. Our approach achieves:

- **98.4% average Defense Success Rate (DSR)** against jailbreak attacks
- **78.4% DSR** against toxic image attacks  
- **100% perfect defense** against gradient-based ImgJP attacks
- **Minimal inference overhead** with competitive utility performance



## ⚠️ Content Warning

The `images/` folder contains test images depicting harmful content including:
- Violence and gore (blood, weapons)
- Adult content (pornography)
- Harmful substances (alcohol, cigarettes)
- Offensive gestures

**Please exercise caution when accessing these files.** They are included solely for research and evaluation purposes.


## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Amadeuszhao/QMLLM.git
cd Q-MLLM

# Install requirements
pip install -r requirements.txt
```

### Usage

```python
from transformers import LlavaProcessor
from unitok_qllava import LlavaForConditionalGeneration
from utils.process import process_with_unitok
from PIL import Image
import torch

# Load model and processor
model_path = "vincentchao/qmllm_unitok"
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).cuda()

processor = LlavaProcessor.from_pretrained(model_path)

# Process an image
image = Image.open("path/to/image.jpg").convert('RGB')

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image"},
        ],
    }, 
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = process_with_unitok(processor, images=image, text=prompt, return_tensors="pt").to(model.device).to(model.dtype)

# Generate response
input_length = inputs['input_ids'].shape[1]

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
    
description = processor.decode(output[0][input_length:], skip_special_tokens=True)
print(description)
```

### Handling Toxic Content Detection

When harmful content is detected, the model raises a `ValueError` with category information:

```python
try:
    # Process image as shown above
    generated_text = processor.decode(output[0][input_length:], skip_special_tokens=True)
    print(f"✓ Image is safe, generated result:\n{generated_text}")
    
except ValueError as e:
    # Extract category information from error message
    error_msg = str(e)
    category_id = int(error_msg.split("category=")[-1].strip())
    category_name = id2class[category_id]
    print(f"⚠️ Harmful content detected!")
    print(f"   Category ID: {category_id}")
    print(f"   Category Name: {category_name}")
```

### Running the Demo

Open and run the Jupyter notebook:

```bash
jupyter notebook test_on_qllava.ipynb
```


## 🧪 Toxic Content Categories

The model can detect and refuse the following categories:

```python
id2class = ['neutral', 'porn', 'gun', 'alcohol', 'blood', 'insulting_gesture', 'cigarette', 'knife']

classes2id = {
    0: 'neutral',
    1: 'porn',
    2: 'gun',
    3: 'alcohol',
    4: 'blood',
    5: 'insulting_gesture',
    6: 'cigarette',
    7: 'knife'
}
```

## 🆕 Updates

### Version 2.0 - UniTok Integration (November 2025)

We are excited to announce **Q-MLLM v2.0**, which incorporates the state-of-the-art **UniTok discrete image encoder** ([arxiv:2502.20321](https://arxiv.org/pdf/2502.20321)) as our image tokenization backbone. This upgrade brings:

- **Enhanced Performance**: Significantly improved multimodal understanding and generation quality
- **Maintained Security**: Same robust defense capabilities (98.4% jailbreak DSR, 78.4% image DSR)
- **Better Efficiency**: Optimized discrete representation learning
- **Stronger Foundation**: Built on UniTok's unified tokenization framework for images and text

The integration of UniTok provides a more powerful discrete bottleneck while preserving Q-MLLM's core security guarantees, demonstrating that safety and performance can advance together.

**Model checkpoint:** `vincentchao/qmllm_unitok`

## 📊 Performance Highlights

| Defense Method | Jailbreak DSR | Image DSR | Overhead |
|----------------|---------------|-----------|----------|
| LLaVA-1.5 | 49.5% | 1.0% | - |
| MLLM-Protector | 91.7% | 53.3% | High |
| ETA | 92.1% | 54.7% | High |
| **Q-MLLM-7B** | **98.4%** | **75.9%** | **Minimal** |
| **Q-MLLM v2.0 (UniTok)** | **98.4%** | **78.4%** | **Minimal** |

*v2.0 maintains identical security performance while significantly improving general task performance through UniTok integration.*

## 🔬 Technical Details

### Two-Level Vector Quantization

1. **Semantic-Level Quantization**: Maps global CLS tokens to discrete semantic embeddings (K=128 codebook)
2. **Patch-Level Quantization**: Discretizes spatial features for fine-grained robustness (P=16000 codebook)

### v2.0: UniTok Integration

The latest version leverages **UniTok** as the discrete image encoder, which provides:
- Unified tokenization framework across modalities
- Advanced vector quantization with improved reconstruction
- Better semantic preservation during discretization
- Enhanced cross-modal alignment capabilities

### Defense Mechanisms

- **Stop-Gradient Operation**: Blocks backpropagation through quantization
- **Discretization Bottleneck**: Forces continuous features into finite discrete space
- **Enhanced CLS Detection**: Improved semantic alignment for toxic content classification



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please open an issue in this repository.

## 📜 License

This project is released for research purposes only. Please use responsibly and ethically.

---

<div align="center">

**⚠️ Ethical Use Notice ⚠️**

This research tool is designed to improve AI safety. The toxic content detection capabilities should only be used for:
- Research purposes
- Safety evaluation
- Model robustness testing

**Do not use this tool to generate, distribute, or process harmful content for malicious purposes.**

</div>
