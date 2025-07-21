import gdown

# Google Drive file ID extracted from your link
file_id = "11GnEWwB0HWViMY2zTOuvgBYDc-MY0-3U"
url = f"https://drive.google.com/uc?id={file_id}"

# Local output filename
output = "unet_epoch_30.pth"

# Download the file
gdown.download(url, output, quiet=False)
