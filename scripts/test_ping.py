from openai import OpenAI

# local vllm routing
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-dummy"
)

# test standard inference
response = client.chat.completions.create(
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the concept of attention mechanisms in two sentences."}
    ],
    max_tokens=100,
    temperature=0.1
)

print(response.choices[0].message.content)