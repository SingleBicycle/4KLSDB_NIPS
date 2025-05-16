import requests
import torch
from transformers import AutoModelForCausalLM
from PIL import Image

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "q-future/one-align", 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Download the image
url = "https://www.clipartbest.com/cliparts/4c9/okk/4c9okkxKi.png"
img = Image.open(requests.get(url, stream=True).raw)
url1 = "https://cdn.shopify.com/s/files/1/0684/0407/products/20190626-6135.jpg?v=1566337499"
img1 = Image.open(requests.get(url1, stream=True).raw)
url2 = "https://www.learnreligions.com/thmb/uUo_cN6g7jaoWwnW9xTX7Q_J48o=/5760x3840/filters:no_upscale():max_bytes(150000):strip_icc()/boy-running-with-gian-snow-ball-636925664-5a54bbe69e94270037d47f0e.jpg"
img2 = Image.open(requests.get(url2, stream=True).raw)

# Change the image resolution to 512x512 (you can adjust the size as needed)
img_resized1 = img.resize((512, 512), Image.LANCZOS)
img_resized2 = img1.resize((512, 512), Image.LANCZOS)
img_resized3 = img2.resize((512, 512), Image.LANCZOS)

# Compute the score using the resized image
score = model.score([img_resized1], task_="quality", input_="image")
score1 = model.score([img_resized2], task_="quality", input_="image")
score2 = model.score([img_resized3], task_="quality", input_="image")
print("image 1:", score, "\nimage 2:", score1, "\nimage 3:", score2)

############################ quality test ###############################################


############################ aesthetics test ############################################
# Download the image
url = "https://www.clipartbest.com/cliparts/4c9/okk/4c9okkxKi.png"
img = Image.open(requests.get(url, stream=True).raw)
url1 = "https://cdn.shopify.com/s/files/1/0684/0407/products/20190626-6135.jpg?v=1566337499"
img1 = Image.open(requests.get(url1, stream=True).raw)
url2 = "https://www.learnreligions.com/thmb/uUo_cN6g7jaoWwnW9xTX7Q_J48o=/5760x3840/filters:no_upscale():max_bytes(150000):strip_icc()/boy-running-with-gian-snow-ball-636925664-5a54bbe69e94270037d47f0e.jpg"
img2 = Image.open(requests.get(url2, stream=True).raw)

# Change the image resolution to 512x512 (you can adjust the size as needed)
img_resized1 = img.resize((512, 512), Image.LANCZOS)
img_resized2 = img1.resize((512, 512), Image.LANCZOS)
img_resized3 = img2.resize((512, 512), Image.LANCZOS)

# Compute the score using the resized image
score = model.score([img_resized1], task_="aesthetics", input_="image")
score1 = model.score([img_resized2], task_="aesthetics", input_="image")
score2 = model.score([img_resized3], task_="aesthetics", input_="image")
print("image 1:", score, "\nimage 2:", score1, "\nimage 3:", score2)



print("without resize")


# Download the image
url = "https://www.clipartbest.com/cliparts/4c9/okk/4c9okkxKi.png"
img = Image.open(requests.get(url, stream=True).raw)
url1 = "https://cdn.shopify.com/s/files/1/0684/0407/products/20190626-6135.jpg?v=1566337499"
img1 = Image.open(requests.get(url1, stream=True).raw)
url2 = "https://www.learnreligions.com/thmb/uUo_cN6g7jaoWwnW9xTX7Q_J48o=/5760x3840/filters:no_upscale():max_bytes(150000):strip_icc()/boy-running-with-gian-snow-ball-636925664-5a54bbe69e94270037d47f0e.jpg"
img2 = Image.open(requests.get(url2, stream=True).raw)


# Compute the score using the resized image
score = model.score([img], task_="aesthetics", input_="image")
score1 = model.score([img1], task_="aesthetics", input_="image")
score2 = model.score([img2], task_="aesthetics", input_="image")
print("image 1:", score, "\nimage 2:", score1, "\nimage 3:", score2)
