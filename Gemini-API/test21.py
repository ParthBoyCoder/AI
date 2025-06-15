import google.generativeai as genai

genai.configure(api_key="KEY")

myfile = genai.upload_file("ti.png")
#print(f"{myfile=}")

model = genai.GenerativeModel("gemini-1.5-flash")
result = model.generate_content(
    [myfile, "\n\n", "What do you see? Ignore the menu and only see the main image in the centre."]
)
print(f"{result.text=}")