import os
from mistralai import Mistral
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

model_name = "mistral-large-latest"
client = Mistral(api_key=api_key)

SYSTEM_PROMPT = """
You are a function-extraction assistant.
User will ask for a plot in free text (e.g. "Plot sin(3x) from -10 to 10").
You must:
1) Figure out the function type from this limited set:
   - polynomial of degree up to 4 (e.g. x^3 - 3x^2 + 5x - 1)
   - sin(k*x)
   - cos(k*x)
   - or simple forms: x, x^2, sin(x), cos(x).
2) Extract the numeric interval [x_min, x_max].
3) Return one line in a pipe-separated format:
   function_name|parameter_list|x_min|x_max
   For example:
   - "polynomial|[1,-3,5,-1]|-2|5" for x^3 - 3x^2 + 5x - 1 on interval [-2,5]
   - "sin|[3]|-10|10" for sin(3x) on interval [-10,10]
   - "x^2|-5|5" for x^2 on interval [-5,5]
4) If the user says something indicating they are done, just return "exit".
Return nothing else, only the exact pipe-separated line or "exit".
"""

def parse_llm_output(llm_text):
    """
    Parses the pipe-separated line from the LLM, e.g.:
      - "polynomial|[1,-3,5,-1]|-2|5"
      - "sin|[3]|-6.28|6.28"
      - "x^2|-5|5"
      - "exit"
    Returns a dictionary with the necessary info, or "exit" for session end.
    """
    llm_text = llm_text.strip().lower()
    if "exit" in llm_text:
        return "exit"

    # Remove extra spaces and newlines
    llm_text = llm_text.replace(" ", "").replace("\n", "")
    parts = llm_text.split("|")  # e.g. ["polynomial","[1,-3,5,-1]","-2","5"]

    print("Split by '|':", parts)

    if len(parts) < 3:
        return None

    function_name = parts[0]

    if function_name == "polynomial":
        # format: polynomial|[1,-3,5,-1]|x_min|x_max
        if len(parts) < 4:
            return None
        param_str = parts[1].strip("[]")  # "1,-3,5,-1"
        coeffs = [float(x) for x in param_str.split(",")] if param_str else []
        x_min = float(parts[2])
        x_max = float(parts[3])
        return {
            "type": "polynomial",
            "coeffs": coeffs,
            "x_min": x_min,
            "x_max": x_max
        }

    elif function_name in ["sin", "cos"]:
        # e.g. sin|[3]|-10|10
        if len(parts) < 4:
            return None
        param_str = parts[1].strip("[]")  # "3"
        k = float(param_str) if param_str else 1.0
        x_min = float(parts[2])
        x_max = float(parts[3])
        return {
            "type": function_name,
            "k": k,
            "x_min": x_min,
            "x_max": x_max
        }

    else:
        # e.g. x^2|-5|5  (should have exactly 3 parts)
        if len(parts) != 3:
            return None
        x_min = float(parts[1])
        x_max = float(parts[2])
        return {
            "type": function_name,
            "x_min": x_min,
            "x_max": x_max
        }

def plot_function(parsed):
    if not parsed or parsed == "exit":
        return

    x_min = parsed["x_min"]
    x_max = parsed["x_max"]
    x = np.linspace(x_min, x_max, 300)
    fn_type = parsed["type"]

    if fn_type == "polynomial":
        coeffs = parsed["coeffs"]
        y = np.polyval(coeffs, x)
        plt.title(f"Polynomial {coeffs}")

    elif fn_type == "sin":
        k = parsed.get("k", 1.0)
        y = np.sin(k * x)
        plt.title(f"y = sin({k}x)")

    elif fn_type == "cos":
        k = parsed.get("k", 1.0)
        y = np.cos(k * x)
        plt.title(f"y = cos({k}x)")

    else:
        # Possibly "x", "x^2", "sin(x)", or "cos(x)" from older version
        if fn_type == "x":
            y = x
            plt.title("y = x")
        elif fn_type == "x^2":
            y = x**2
            plt.title("y = x^2")
        elif fn_type == "sin(x)":
            y = np.sin(x)
            plt.title("y = sin(x)")
        elif fn_type == "cos(x)":
            y = np.cos(x)
            plt.title("y = cos(x)")
        else:
            y = x
            plt.title("Unrecognized. Plotting y = x as fallback.")

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

def main():
    session_active = True

    while session_active:
        user_input = input("User: ")

        if any(word in user_input.lower() for word in ["bye", "exit", "done", "quit", "stop"]):
            print("User ended the session.")
            break

        response = client.chat.complete(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )

        llm_answer = response.choices[0].message.content.strip()
        print(f"LLM raw output: {llm_answer}")

        parsed = parse_llm_output(llm_answer)
        print("parsed:", parsed)

        if parsed == "exit":
            print("LLM indicated the session has ended.")
            session_active = False
        elif parsed:
            plot_function(parsed)
        else:
            print("Could not parse the LLM response. Please try again.")

    print("Session closed.")

if __name__ == "__main__":
    main()