import gdown

file_id = "1TfUFre0Cc-dliH65eVFSrLdFWbEU4Dc8"
url = f"https://drive.google.com/uc?id={file_id}"
output = "project.zip"

gdown.download(url, output, quiet=False)
