# mask-detector
During Covid19 quarantine i decided to implement my own version of face mask detector using OpenCV, Keras and Tensorflow Deep Learning models just for fun.

[![Alt text](https://img.youtube.com/vi/GhPpzQR-WAI/0.jpg)](https://www.youtube.com/watch?v=GhPpzQR-WAI)


### Checkout the code:
```
git clone https://github.com/fnandocontreras/mask-detector.git
```

### Install Requirements

To be able to run the code, before you need to install the required packages:

To install using anaconda run the following commands:

```
conda env create -f environment.yml
conda activate maskdetector
```
The file "environment.yml" contains the required packages for running mask-detector and after running the commands above, anaconda will create a virtual environment name "maskdetector" with all the requirements specified in the env file. If you don't have anaconda you can install the required packages by hand using PIP:
```
pip install numpy tensorflow keras mtcnn pillow opencv
```

After installation you are ready to run the mask detector program

### Running the mask detector

For running the mask detector using your webcam live streaming run the following command:
```
python app.py
```


For running the mask detector on a video file run the following command:
```
python app.py --source <your-video-filename>
```

For some videos made with a smartphone you can experience that the image is rotated 90 degrees, this is because the mask detector is not checking orientation. To fix this you can pass the argument: "rotate" like:

```
python app.py --source <your-video-filename> --rotate
```



### Recording

You can also record the video streaming by passing the argument "--rec" like:

```
python app.py --rec
```
It will record the video streaming in a video file named: "record.avi"


### Training the classification model

If you are curious to see how i trained the model using tensorflow and keras you can see the notebook: [training_model.ipynb](training_model.ipynb)
