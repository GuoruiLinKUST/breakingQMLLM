# breakingQMLLM
Fix gradient obfuscation in "Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security"

# Installation

```commandline
conda create -n some_name python=3.11
conda activate some_name
pip install -r requirements.txt
```

# Reproducing Our Results

If you have plenty of time, run ```breakingDefense.py``` to generate adversarial examples and jailbreak the model. 
We also provide adversarial examples in ```./fig```. You can run ```breakingDefenseEval.py``` to directly jailbreak the model with our adversarial examples.
