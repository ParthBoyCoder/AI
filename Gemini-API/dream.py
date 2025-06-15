import google.generativeai as genai
import pyautogui

pyautogui.screenshot("S1.png")


genai.configure(api_key="KEY")

myfile = genai.upload_file("S1.png")
#print(f"{myfile=}")

model = genai.GenerativeModel("gemini-1.5-flash")
result = model.generate_content(
    [myfile, "\n\n", "Focus on the quiz question and tell the answer only"]
)
print(result.text)