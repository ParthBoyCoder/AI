import google.generativeai as genai

genai.configure(api_key="KEY")

myfile = genai.upload_file("Cajun_instruments.jpg")
#print(f"{myfile=}")

model = genai.GenerativeModel("gemini-1.5-flash")
result = model.generate_content(
    [myfile, "\n\n", "Can you tell me about the instruments in this photo?"]
)
print(f"{result.text=}")