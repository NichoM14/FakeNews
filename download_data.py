import kagglehub
def main():

# Download latest version
    path = kagglehub.dataset_download("doanquanvietnamca/liar-dataset")

    print("Path to dataset files:", path)


if __name__ == "__main__":
    main()
