import requests

img="https://img.logo.dev/"+input("Cpmpany URL: ")+"?token=pk_WeNuh2LWTt-yrDHvQJRAMA"

response = requests.get(img)
if response.status_code == 200:
    with open("logo.png", "wb") as file:
        file.write(response.content)
    print("Logo downloaded successfully as logo.png")
