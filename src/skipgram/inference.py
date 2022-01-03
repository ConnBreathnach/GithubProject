from tensorflow.keras.models import load_model

model = load_model('models/skipgram_model.h5')

print(model.summary())