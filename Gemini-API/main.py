import google.generativeai as genai
import pyautogui
import pygetwindow as gw

# Get the Chrome window
chrome_windows = [window for window in gw.getWindowsWithTitle('Chrome') if 'Chrome' in window.title]

if chrome_windows:
    chrome_window = chrome_windows[0]
    chrome_window.activate()

    # Get the bounding box of the Chrome window
    left, top, right, bottom = chrome_window.left, chrome_window.top, chrome_window.right, chrome_window.bottom

    # Take the screenshot
    screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))

    # Save the screenshot
    screenshot.save('screenshot.png')
else:
    print("No Chrome window found.")


genai.configure(api_key="KEY")

myfile = genai.upload_file("screenshot.png")
#print(f"{myfile=}")

model = genai.GenerativeModel("gemini-1.5-flash")
result = model.generate_content(
    [myfile, "\n\n", "What do you see? Ignore the menu and only see the main image in the centre."]
)
print(f"{result.text=}")