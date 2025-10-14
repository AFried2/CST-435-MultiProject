# convert_model.py
import tensorflow as tf

print("Loading the full model from natural_scene_classifier.keras...")
# Load the model you already trained
model = tf.keras.models.load_model('natural_scene_classifier.keras')

print("Saving model weights to model_weights.weights.h5...")
# Save just the weights into a new file
model.save_weights('model_weights.weights.h5')

print("Conversion complete!")