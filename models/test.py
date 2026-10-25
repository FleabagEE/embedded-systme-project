import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="models/1s_2s/DSCNN_64_SC_all/20261023T213435Z_774e54f8_ffb532/subnets/DSCNN_64_SC_1/exports/1s/int8.tflite"
)
interpreter.allocate_tensors()

print("Input:", interpreter.get_input_details()[0]["dtype"])
print("Output:", interpreter.get_output_details()[0]["dtype"])

print("\nOperators:")
for op in interpreter._get_ops_details():
    print(op["op_name"])

print("\nTensor data types:")
for tensor in interpreter.get_tensor_details():
    print(tensor["name"], tensor["dtype"])