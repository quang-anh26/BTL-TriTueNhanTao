import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
import zipfile, pickle

def load_pt_params(path):
    with zipfile.ZipFile(path, 'r') as z:
        prefix = 'best'
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
                if module in ['collections', 'builtins']: return getattr(__import__(module), name)
                if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                    def rebuild(storage_info, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
                        key = storage_info[1]
                        arr = storages[key]
                        sub = arr[offset : offset + int(np.prod(size))]
                        return sub.reshape(size)
                    return rebuild
                if module == 'torch._utils' and name == '_rebuild_parameter':
                    return lambda d, *a: d
                class Stub:
                    full_name = f'{module}.{name}'
                    def __init__(self, *a, **k): pass
                    def __setstate__(self, s):
                        if isinstance(s, dict): self.__dict__.update(s)
                        elif isinstance(s, tuple) and len(s) > 1 and isinstance(s[1], dict):
                            self.__dict__.update(s[1])
                return Stub
        with z.open(f'{prefix}/data.pkl') as f:
            data = Unpickler(f).load()
        m = data['model']
        params = {}
        def walk(obj, p=''):
            if hasattr(obj, '_parameters'):
                for k, v in obj._parameters.items():
                    if isinstance(v, np.ndarray): params[f'{p}{k}'] = v
            if hasattr(obj, '_buffers'):
                for k, v in obj._buffers.items():
                    if isinstance(v, np.ndarray): params[f'{p}{k}'] = v
            if hasattr(obj, '_modules'):
                for k, v in obj._modules.items():
                    walk(v, f'{p}{k}.')
        walk(m)
        return params

def fuse_conv_bn(w, bn_w, bn_b, bn_m, bn_v, eps=1e-5):
    scale = bn_w / np.sqrt(bn_v + eps)
    fused_w = w * scale.reshape(-1, 1, 1, 1)
    fused_b = bn_b - bn_m * scale
    return fused_w.astype(np.float32), fused_b.astype(np.float32)

def build_plate_onnx():
    pt_params = load_pt_params('4_backend/best.pt')
    model = onnx.load('yolov8n.onnx')

    # Build fused weights for all Conv+BN in best.pt
    fused_weights = {}
    fused_biases = {}
    for k in list(pt_params.keys()):
        if k.endswith('.conv.weight'):
            base = k[:-len('.conv.weight')]
            if f'{base}.bn.weight' in pt_params:
                w = pt_params[k]
                bn_w = pt_params[f'{base}.bn.weight']
                bn_b = pt_params[f'{base}.bn.bias']
                bn_m = pt_params[f'{base}.bn.running_mean']
                bn_v = pt_params[f'{base}.bn.running_var']
                fw, fb = fuse_conv_bn(w, bn_w, bn_b, bn_m, bn_v)
                fused_weights[k] = fw
                fused_biases[f'{base}.conv.bias'] = fb

    # Update initializers
    new_inits = []
    for init in model.graph.initializer:
        name = init.name
        if name in fused_weights:
            new_inits.append(numpy_helper.from_array(fused_weights[name], name=name))
        elif name in fused_biases:
            new_inits.append(numpy_helper.from_array(fused_biases[name], name=name))
        elif name in pt_params:
            # Unfused weights (e.g. cv2.*.2.weight, cv3.*.2.weight, dfl.conv.weight)
            val = pt_params[name].astype(np.float32)
            new_inits.append(numpy_helper.from_array(val, name=name))
        else:
            # Constants or graph constants
            arr = numpy_helper.to_array(init)
            # Check if this is Split_388: [64, 80] -> [64, 1]
            if name == 'onnx::Split_388' or (arr.ndim == 1 and len(arr) == 2 and arr[0] == 64 and arr[1] == 80):
                new_inits.append(numpy_helper.from_array(np.array([64, 1], dtype=arr.dtype), name=name))
            # Check if this is Constant_0/1/2 for Reshape: [1, 144, 6400], etc.
            elif arr.ndim == 1 and len(arr) == 3 and arr[1] == 144:
                new_arr = arr.copy()
                new_arr[1] = 65 # 64 bbox + 1 class
                new_inits.append(numpy_helper.from_array(new_arr, name=name))
            else:
                new_inits.append(init)

    # Replace model.graph.initializer
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)

    # Update Constant nodes in graph
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    arr = numpy_helper.to_array(attr.t)
                    if arr.ndim == 1 and len(arr) == 2 and arr[0] == 64 and arr[1] == 80:
                        new_t = numpy_helper.from_array(np.array([64, 1], dtype=arr.dtype))
                        attr.t.CopyFrom(new_t)
                    elif arr.ndim == 1 and len(arr) == 3 and arr[0] == 1 and arr[1] == 144 and arr[2] == -1:
                        new_t = numpy_helper.from_array(np.array([1, 65, -1], dtype=arr.dtype))
                        attr.t.CopyFrom(new_t)

    # Update output0 shape from [1, 84, 8400] to [1, 5, 8400]
    for out in model.graph.output:
        if out.name == 'output0':
            out.type.tensor_type.shape.dim[1].dim_value = 5 # 4 bbox + 1 class

    # Set ir_version to 10 for broad compatibility
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, '4_backend/plate_detector.onnx')
    print('Saved 4_backend/plate_detector.onnx successfully!')

if __name__ == '__main__':
    build_plate_onnx()
