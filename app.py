import google.generativeai as genai

genai.configure(api_key='AIzaSyAuIR4JR9V561N4N7SM9Ti0FAI3-ddEnII')

sample_file = genai.upload_file(path="maths.png", display_name="Maths Question")
file = genai.get_file(name=sample_file.name)

model = genai.GenerativeModel(model_name="gemini-1.5-pro")
response = model.generate_content([sample_file, "Solve the following maths question."])
print(">" + response.text)
