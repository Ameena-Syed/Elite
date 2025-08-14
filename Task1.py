from transformers import pipeline

# Load summarizer
summarizer = pipeline("summarization")

# Read from file
with open("input.txt", "r", encoding="utf-8") as f:
    input_text = f.read()

# Generate summary
summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)
result = summary[0]['summary_text']

# Display summary
print("\n--- Summarized Text ---\n")
print(result)
print("\n")