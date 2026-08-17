import kagglehub
# Download latest version
path = kagglehub.dataset_download("emmarex/plantdisease")
print("Path to dataset files:", path)