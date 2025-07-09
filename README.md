# NCU Smart LLM (phi2-ncu) — Smart LLM Fine-tuned for NCU Tasks

<p align="center">
  <img src="https://huggingface.co/pranav2711/phi2-ncu-model/resolve/main/NCU-Logo.png" alt="NCU Logo" width="350"/>
</p>

> A lightweight, instruction-tuned version of [Microsoft's Phi-2](https://huggingface.co/microsoft/phi-2), customized for use cases and conversations related to The NorthCap University (NCU), India.
> Fine-tuned using LoRA on 1,098 high-quality examples, it's optimized for academic, administrative, and smart campus queries.

---

## Highlights

* **Base Model:** `microsoft/phi-2` (2.7B parameters)
* **Fine-tuned Using:** Low-Rank Adaptation (LoRA) + PEFT + Hugging Face Transformers
* **Dataset:** University questions, FAQs, policies, academic support queries, smart campus data
* **Training Environment:** Google Colab (T4 GPU), 4 epochs, batch size 1, no FP16
* **Final Format:** Full model weights (`.safetensors`) + tokenizer

---

## Model Access

| Platform           | Access Method                                                               |
| ------------------ | --------------------------------------------------------------------------- |
| Hugging Face       | [phi2-ncu-model](https://huggingface.co/pranav2711/phi2-ncu-model)          |
| Hugging Face Space | [Live Chatbot Demo](https://huggingface.co/spaces/pranav2711/phi2-ncu-chat-space) |
| Ollama (Offline)   | `ollama create phi2-ncu -f Modelfile` *(self-hosted only)*                  |

---

## Phi2B - How to Use Locally (Hugging Face Transformers)

```bash
pip install transformers accelerate peft
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load adapter config
adapter_path = "pranav2711/phi2-ncu-model"
base_model = "microsoft/phi-2"

# Load tokenizer and base
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

# Load fine-tuned adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Inference
input_prompt = "### Question:\nHow can I apply for re-evaluation at NCU?\n\n### Answer:"
inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
---

## Mistral 7B - How to Use Locally (Hugging Face Transformers)

```bash
pip install transformers accelerate
```
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("pranav2711/ncu-smartllm-fp32", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("pranav2711/ncu-smartllm-fp32")

prompt = "### Question:\nHow do I apply for hostel at NCU?\n\n### Answer:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Example Dataset Format (Used for Training)

```json
{
  "instruction": "How do I get my degree certificate?",
  "input": "I'm a 2023 BTech passout from CSE at NCU.",
  "output": "You can collect your degree certificate from the admin block on working days between 9AM and 4PM. Carry a valid ID proof."
}
```

Formatted as:

```
### Question:
How do I get my degree certificate?
I'm a 2023 BTech passout from CSE at NCU.

### Answer:
You can collect your degree certificate...
```

---

## Training Strategy

* Used `LoRA` with rank=8, alpha=16
* Tokenized to max length = 512
* Used `Trainer` with `fp16=False` to avoid CUDA AMP issues
* Batch size = 1, Epochs = 4
* Trained on Google Colab (T4), saving final full weights

---

## License

[Apache 2.0](https://huggingface.co/pranav2711/phi2-ncu-model/resolve/main/LICENSE)

## About NCU

**The NorthCap University**, Gurugram (formerly ITM University), is a multidisciplinary university with programs in engineering, management, law, and sciences.

This model was created as part of a research initiative to explore AI for academic services, campus automation, and local LLM deployments.

## Contribute

Have better FAQs or data? Want to train on your college corpus? Fork the repo or raise a PR at:

👉 [https://github.com/pranav2711/ncu-smartllm](https://github.com/pranav2711/ncu-smartllm)
