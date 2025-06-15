import google.generativeai as genai

genai.configure(api_key="KEY")

myfile = genai.upload_file("audio.mp3")
#print(f"{myfile=}")

model = genai.GenerativeModel("gemini-1.5-flash")
result = model.generate_content(
    [myfile, "\n\n", "Tell me everything you can understand from this audio"]
)
print(f"{result.text=}")