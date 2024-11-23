import time
begin = time.time()

#FROM https://huggingface.co/HuggingFaceTB/SmolLM-135M

from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "HuggingFaceTB/SmolLM-135M"
device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

import torch

# Define the movie review for classification
example_review = "This movie was an absolute masterpiece with stunning visuals and a gripping story!"
example_review_neg = "This movie was terrible and I hated it."
example_negative_review_2 = "I really think this movie is not that good. It was a waste of time."

example_inference_review = "what a movie ! changed my life! I love luke's character and actor"
# Prompt for zero-shot classification
prompt = [
    ['Review :', example_review, ' Sentiment[positive/negative] :', ' positive','\n'],
    ['Review :', example_review_neg, ' Sentiment[positive/negative] :', ' negative','\n'],
    ['Review :', example_negative_review_2, ' Sentiment[positive/negative] :', ' negative','\n'],
    ['Review :', example_inference_review, ' Sentiment[positive/negative] :'],

]

prompt_=[]
for i in prompt:
    prompt_.append(''.join(i))
#make all attached
prompt = ''.join(prompt_)

# Tokenize and set up for inference
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate the output
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=5)

# Decode and print the output
output_str = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_str)



end = time.time()
print(f"Inference time: {end - begin:.2f} seconds")