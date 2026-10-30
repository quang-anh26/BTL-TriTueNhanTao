import zipfile, pickle, collections
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import cv2

CHARS = "0123456789ABCDEFGHIKLMNPRSTUVXYZ-"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

def load_crnn_state_dict(pt_path):
    with zipfile.ZipFile(pt_path, 'r') as z:
        prefix = 'best_crnn'
        storages = {}
        class Unpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                type_name = pid[1].full_name
                key = str(pid[2])
                numel = pid[4]
                if key not in storages:
                    dtype = np.float32 if 'Float' in type_name else (np.int64 if 'Long' in type_name else np.float16)
                    raw = z.read(f'{prefix}/data/{key}')
                    storages[key] = np.frombuffer(raw, dtype=dtype)[:numel]
                return ('STORAGE', key)
            def find_class(self, module, name):
                if module == 'collections' and name == 'OrderedDict':
                    return collections.OrderedDict
                if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                    def rebuild(storage_info, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
                        key = storage_info[1]
                        arr = storages[key]
                        sub = arr[offset : offset + int(np.prod(size))]
                        return sub.reshape(size)
                    return rebuild
                class Stub:
                    full_name = f'{module}.{name}'
                    def __init__(self, *a, **k): pass
                    def __setstate__(self, s): pass
                return Stub
        with z.open(f'{prefix}/data.pkl') as f:
            data = Unpickler(f).load()
        return data

def build_crnn_onnx(weights, output_path, img_height=48, img_width=192):
    nodes = []
    initializers = []

    # Input: [1, 1, img_height, img_width]
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, img_height, img_width])
    current_tensor = 'input'

    # 7 Conv blocks + 1 final Conv
    # cnn.0: conv_block(1, 64, pool="both")
    # cnn.1: conv_block(64, 128, pool="both")
    # cnn.2: conv_block(128, 256, pool=None)
    # cnn.3: conv_block(256, 256, pool="height")
    # cnn.4: conv_block(256, 512, pool=None)
    # cnn.5: conv_block(512, 512, pool="height")
    # cnn.6: conv_block(512, 512, pool=None)
    # cnn.7: nn.Conv2d(512, 512, kernel_size=(img_height // 16, 1))

    blocks = [
        (0, "both"),
        (1, "both"),
        (2, None),
        (3, "height"),
        (4, None),
        (5, "height"),
        (6, None),
    ]

    for b_idx, pool in blocks:
        prefix = f'cnn.{b_idx}.'
        # Conv
        w = weights[f'{prefix}0.weight']
        b = weights[f'{prefix}0.bias']
        w_init = numpy_helper.from_array(w, name=f'{prefix}conv_w')
        b_init = numpy_helper.from_array(b, name=f'{prefix}conv_b')
        initializers.extend([w_init, b_init])
        conv_out = f'{prefix}conv_out'
        nodes.append(helper.make_node(
            'Conv', inputs=[current_tensor, f'{prefix}conv_w', f'{prefix}conv_b'],
            outputs=[conv_out], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        ))

        # BatchNorm
        gamma = weights[f'{prefix}1.weight']
        beta = weights[f'{prefix}1.bias']
        mean = weights[f'{prefix}1.running_mean']
        var = weights[f'{prefix}1.running_var']
        initializers.extend([
            numpy_helper.from_array(gamma, name=f'{prefix}bn_gamma'),
            numpy_helper.from_array(beta, name=f'{prefix}bn_beta'),
            numpy_helper.from_array(mean, name=f'{prefix}bn_mean'),
            numpy_helper.from_array(var, name=f'{prefix}bn_var'),
        ])
        bn_out = f'{prefix}bn_out'
        nodes.append(helper.make_node(
            'BatchNormalization',
            inputs=[conv_out, f'{prefix}bn_gamma', f'{prefix}bn_beta', f'{prefix}bn_mean', f'{prefix}bn_var'],
            outputs=[bn_out], epsilon=1e-5
        ))

        # ReLU
        relu_out = f'{prefix}relu_out'
        nodes.append(helper.make_node('Relu', inputs=[bn_out], outputs=[relu_out]))
        current_tensor = relu_out

        # Pool
        if pool == "both":
            pool_out = f'{prefix}pool_out'
            nodes.append(helper.make_node('MaxPool', inputs=[current_tensor], outputs=[pool_out], kernel_shape=[2, 2], strides=[2, 2]))
            current_tensor = pool_out
        elif pool == "height":
            pool_out = f'{prefix}pool_out'
            nodes.append(helper.make_node('MaxPool', inputs=[current_tensor], outputs=[pool_out], kernel_shape=[2, 1], strides=[2, 1]))
            current_tensor = pool_out

    # Final Conv: nn.Conv2d(512, 512, kernel_size=(img_height // 16, 1))
    kh = img_height // 16
    w7 = weights['cnn.7.weight']
    b7 = weights['cnn.7.bias']
    initializers.extend([
        numpy_helper.from_array(w7, name='cnn_7_w'),
        numpy_helper.from_array(b7, name='cnn_7_b'),
    ])
    cnn7_out = 'cnn_7_out'
    nodes.append(helper.make_node('Conv', inputs=[current_tensor, 'cnn_7_w', 'cnn_7_b'], outputs=[cnn7_out], kernel_shape=[kh, 1], pads=[0, 0, 0, 0]))

    # feat shape is [1, 512, 1, 48].
    # In PyTorch:
    # feat = feat.squeeze(2)  -> [1, 512, 48]
    # feat = feat.permute(2, 0, 1) -> [48, 1, 512]
    # In ONNX: Squeeze axes=[2] -> [1, 512, 48], Transpose perm=[2, 0, 1] -> [48, 1, 512]
    squeeze_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name='sq_axes')
    initializers.append(squeeze_axes)
    nodes.append(helper.make_node('Squeeze', inputs=[cnn7_out, 'sq_axes'], outputs=['feat_sq']))
    nodes.append(helper.make_node('Transpose', inputs=['feat_sq'], outputs=['feat_perm'], perm=[2, 0, 1]))
    current_tensor = 'feat_perm' # [48, 1, 512]

    # Convert PyTorch LSTM weights to ONNX LSTM format
    def convert_lstm_weights(rnn_prefix, input_size, hidden_size):
        # PyTorch gates: (i, f, g, o) -> 0: i, 1: f, 2: g, 3: o
        # ONNX gates: (i, o, f, c) -> 0: i, 1: o, 2: f, 3: g
        order = [0, 3, 1, 2] # slice indices
        def reorder(gate_tensor, H):
            gates = [gate_tensor[idx * H : (idx + 1) * H] for idx in order]
            return np.concatenate(gates, axis=0)

        # Forward
        w_ih_f = weights[f'{rnn_prefix}.weight_ih_l0']
        w_hh_f = weights[f'{rnn_prefix}.weight_hh_l0']
        b_ih_f = weights[f'{rnn_prefix}.bias_ih_l0']
        b_hh_f = weights[f'{rnn_prefix}.bias_hh_l0']

        # Reverse
        w_ih_r = weights[f'{rnn_prefix}.weight_ih_l0_reverse']
        w_hh_r = weights[f'{rnn_prefix}.weight_hh_l0_reverse']
        b_ih_r = weights[f'{rnn_prefix}.bias_ih_l0_reverse']
        b_hh_r = weights[f'{rnn_prefix}.bias_hh_l0_reverse']

        W = np.stack([reorder(w_ih_f, hidden_size), reorder(w_ih_r, hidden_size)], axis=0)
        R = np.stack([reorder(w_hh_f, hidden_size), reorder(w_hh_r, hidden_size)], axis=0)
        B_f = np.concatenate([reorder(b_ih_f, hidden_size), reorder(b_hh_f, hidden_size)], axis=0)
        B_r = np.concatenate([reorder(b_ih_r, hidden_size), reorder(b_hh_r, hidden_size)], axis=0)
        B = np.stack([B_f, B_r], axis=0)
        return W.astype(np.float32), R.astype(np.float32), B.astype(np.float32)

    # RNN 0: BidirectionalLSTM(512, hidden_size=256, output_size=256)
    W0, R0, B0 = convert_lstm_weights('rnn.0.rnn', 512, 256)
    initializers.extend([
        numpy_helper.from_array(W0, name='rnn0_W'),
        numpy_helper.from_array(R0, name='rnn0_R'),
        numpy_helper.from_array(B0, name='rnn0_B'),
    ])
    nodes.append(helper.make_node(
        'LSTM', inputs=[current_tensor, 'rnn0_W', 'rnn0_R', 'rnn0_B'],
        outputs=['rnn0_Y'], direction='bidirectional', hidden_size=256
    ))
    # rnn0_Y shape: [48, 2, 1, 256] -> [seq_len, num_directions, batch_size, hidden_size]
    # In PyTorch, BiLSTM forward returns: out of shape [seq_len, batch_size, 2 * hidden_size]
    # In ONNX: Transpose [0, 2, 1, 3] -> [48, 1, 2, 256], Reshape -> [48, 1, 512]
    nodes.append(helper.make_node('Transpose', inputs=['rnn0_Y'], outputs=['rnn0_Y_tr'], perm=[0, 2, 1, 3]))
    reshape_shape0 = numpy_helper.from_array(np.array([48, 1, 512], dtype=np.int64), name='rs_shape0')
    initializers.append(reshape_shape0)
    nodes.append(helper.make_node('Reshape', inputs=['rnn0_Y_tr', 'rs_shape0'], outputs=['rnn0_Y_flat']))

    # Linear: rnn.0.fc (512 -> 256)
    fc0_w = weights['rnn.0.fc.weight'] # (256, 512)
    fc0_b = weights['rnn.0.fc.bias']   # (256,)
    initializers.extend([
        numpy_helper.from_array(fc0_w.T, name='fc0_w'),
        numpy_helper.from_array(fc0_b, name='fc0_b'),
    ])
    # MatMul + Add
    nodes.append(helper.make_node('MatMul', inputs=['rnn0_Y_flat', 'fc0_w'], outputs=['fc0_mm']))
    nodes.append(helper.make_node('Add', inputs=['fc0_mm', 'fc0_b'], outputs=['fc0_out']))
    current_tensor = 'fc0_out' # [48, 1, 256]

    # RNN 1: BidirectionalLSTM(256, hidden_size=256, output_size=NUM_CLASSES)
    W1, R1, B1 = convert_lstm_weights('rnn.1.rnn', 256, 256)
    initializers.extend([
        numpy_helper.from_array(W1, name='rnn1_W'),
        numpy_helper.from_array(R1, name='rnn1_R'),
        numpy_helper.from_array(B1, name='rnn1_B'),
    ])
    nodes.append(helper.make_node(
        'LSTM', inputs=[current_tensor, 'rnn1_W', 'rnn1_R', 'rnn1_B'],
        outputs=['rnn1_Y'], direction='bidirectional', hidden_size=256
    ))
    nodes.append(helper.make_node('Transpose', inputs=['rnn1_Y'], outputs=['rnn1_Y_tr'], perm=[0, 2, 1, 3]))
    reshape_shape1 = numpy_helper.from_array(np.array([48, 1, 512], dtype=np.int64), name='rs_shape1')
    initializers.append(reshape_shape1)
    nodes.append(helper.make_node('Reshape', inputs=['rnn1_Y_tr', 'rs_shape1'], outputs=['rnn1_Y_flat']))

    # Linear: rnn.1.fc (512 -> NUM_CLASSES=35)
    fc1_w = weights['rnn.1.fc.weight'] # (NUM_CLASSES, 512)
    fc1_b = weights['rnn.1.fc.bias']   # (NUM_CLASSES,)
    initializers.extend([
        numpy_helper.from_array(fc1_w.T, name='fc1_w'),
        numpy_helper.from_array(fc1_b, name='fc1_b'),
    ])
    nodes.append(helper.make_node('MatMul', inputs=['rnn1_Y_flat', 'fc1_w'], outputs=['fc1_mm']))
    nodes.append(helper.make_node('Add', inputs=['fc1_mm', 'fc1_b'], outputs=['output']))

    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [48, 1, NUM_CLASSES])

    graph = helper.make_graph(nodes, 'crnn', [input_info], [output_info], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved {output_path} successfully!")

if __name__ == '__main__':
    w_one = load_crnn_state_dict('3_recognition/runs/recognition_one_row_v2/best_crnn.pt')
    build_crnn_onnx(w_one, '4_backend/crnn_one_row.onnx')
    w_two = load_crnn_state_dict('3_recognition/runs/recognition_two_row_v2/best_crnn.pt')
    build_crnn_onnx(w_two, '4_backend/crnn_two_row.onnx')

