import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="wakeword_models/hey_co_pilot/hey_co_pilot.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Inputs:", input_details)
print("Outputs:", output_details)

# Optionally, you can get the intermediate tensor details
for detail in input_details:
    print("Tensor name:", detail['name'])
    print("Shape:", detail['shape'])
    print("Type:", detail['dtype'])
