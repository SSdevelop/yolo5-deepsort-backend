# Airport Project Backend

### Installation guide

1. Make sure you have docker installed and running.
2. Navigate to the project folder.
3. Run the following command in your terminal:
```sh
docker build -t airport_backend .
```
4. Once the image is built, run the image using the command given below in your terminal. This above command will mount your project directory into your container. So, any change will be reflected to the container directly.
```sh
docker run -it --rm -p 0.0.0.0:5000:5000 -v .:/app --gpus all --name airport_backend airport_backend bash ./run.sh
```
5. Run the frontend seperatly and test.