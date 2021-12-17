import tensorflow as tf
import numpy as np
from official.nlp.data import classifier_data_lib as cdl
from official.nlp.bert import tokenization
from tensorflow_hub import registry
import os

DEFAULT_SCALE, DEFAULT_ZERO_POINT = 0, 0

uri = "https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/1"
vocab_file = os.path.join(
    registry.resolver(uri), 'assets', 'vocab.txt')

tokenizer = tokenization.FullTokenizer(vocab_file, True)


def to_feature(text, label=None, label_list=None, max_seq_length=128, tokenizer=tokenizer):

    example = cdl.InputExample(guid=None,
                            text_a=text,
                            text_b=None,
                            label=label)
    feature = cdl.convert_single_example(
        0, example, label_list, max_seq_length, tokenizer)

    return (np.array(feature.input_ids), np.array(feature.input_mask), np.array(feature.segment_ids))


def _get_input_tensor(input_tensors, input_details, i):
    """Gets input tensor in `input_tensors` that maps `input_detail[i]`."""
    if isinstance(input_tensors, dict):
        # Gets the mapped input tensor.
        input_detail = input_details[i]
        for input_tensor_name, input_tensor in input_tensors.items():
            if input_tensor_name in input_detail['name']:
                return input_tensor
        raise ValueError('Input tensors don\'t contains a tensor that mapped the '
                        'input detail %s' % str(input_detail))
    else:
        return input_tensors[i]


class LiteRunner(object):


    def __init__(self,tflite_filepath):

        with tf.io.gfile.GFile(tflite_filepath, 'rb') as f:
            tflite_model = f.read()
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()

        # Gets the indexed of the input tensors.
        self.input_details = self.interpreter.get_input_details()


        self.output_details = self.interpreter.get_output_details()


    def run(self, input_tensors):

        if not isinstance(input_tensors, list) and \
                not isinstance(input_tensors, tuple) and \
                not isinstance(input_tensors, dict):
            input_tensors = [input_tensors]

        interpreter = self.interpreter

        # Reshape inputs
        for i, input_detail in enumerate(self.input_details):
            input_tensor = _get_input_tensor(input_tensors, self.input_details, i)
            interpreter.resize_tensor_input(input_detail['index'], input_tensor.shape)
        interpreter.allocate_tensors()

        # Feed input to the interpreter
        for i, input_detail in enumerate(self.input_details):
            input_tensor = _get_input_tensor(input_tensors, self.input_details, i)
            if input_detail['quantization'] != (DEFAULT_SCALE, DEFAULT_ZERO_POINT):
                # Quantize the input
                scale, zero_point = input_detail['quantization']
                input_tensor = input_tensor / scale + zero_point
                input_tensor = np.array(input_tensor, dtype=input_detail['dtype'])
            interpreter.set_tensor(input_detail['index'], input_tensor.astype(np.int32))

        interpreter.invoke()

        output_tensors = []
        for output_detail in self.output_details:
            output_tensor = interpreter.get_tensor(output_detail['index'])
            if output_detail['quantization'] != (DEFAULT_SCALE, DEFAULT_ZERO_POINT):
                # Dequantize the output
                scale, zero_point = output_detail['quantization']
                output_tensor = output_tensor.astype(np.float32)
                output_tensor = (output_tensor - zero_point) * scale
            output_tensors.append(output_tensor)

        if len(output_tensors) == 1:
            return output_tensors[0]
        return output_tensors


def predict(text):
    encoded = to_feature(text)

    runner = LiteRunner('lite/model.tflite')

    preds = runner.run(encoded)

    pred = np.argmax(preds)
    
    if pred == 0:
        return 'Non-sucide'
    else:
        return 'Sucide'