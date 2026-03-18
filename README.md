# CNN Clothing Classifier

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify clothing items from the Fashion MNIST dataset into 10 categories. Includes a Streamlit web app for real-time image classification.

## Demo
🔗 [Live App](#) *(coming soon)*

## Categories
The model classifies images into:
`T-shirt/top` · `Trouser` · `Pullover` · `Dress` · `Coat` · `Sandal` · `Shirt` · `Sneaker` · `Bag` · `Ankle boot`

## Tech Stack
- Python 3.11
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Docker

## Run Locally
```bash
git clone https://github.com/Aarushi-Bhargava/cnn-clothing-classifier.git
cd cnn-clothing-classifier
pip install -r app/requirements.txt
streamlit run app/main.py
```

## Run with Docker
```bash
docker build -t cnn-clothing-classifier .
docker run -p 8501:8501 cnn-clothing-classifier
```