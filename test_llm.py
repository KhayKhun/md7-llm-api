import os
import subprocess
from mistralai import Mistral
from dotenv import load_dotenv

isEnd = False


def extract_and_run_code(input_str):
    global isEnd
    start = input_str.find("```python\n")
    if start == -1:
        isEnd = True
        return
    count = 0
    while start != -1:
        count += 1
        end = input_str.find("```", start + 3)
        if end == -1:
            break
        code = input_str[start + 9:end].strip()  # Adjust to remove "```python"
        print("---code start-------")
        print(code)
        print("----code end------")

        with open("temp.py", "w") as f:
            f.write(code)
        try:
            subprocess.run(["python3", "temp.py"])
        except Exception as e:
            print("Error:", e)
        start = input_str.find("```python", end + 3)  # Find next code block
    print(count)


# no touch
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"
client = Mistral(api_key=api_key)
# no touch

while not isEnd:
    content = input("Enter a prompt: ")
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ]
    )
    response_content = chat_response.choices[0].message.content
    print("----response start------")
    print(response_content)
    print("-----response end-----")
    extract_and_run_code(response_content)
