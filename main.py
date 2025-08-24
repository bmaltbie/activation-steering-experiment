from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B" # https://huggingface.co/Qwen/Qwen3-8B

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)