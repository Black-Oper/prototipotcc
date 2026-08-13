import onnx

model = onnx.load("../inference-cpp/rtdvsr.onnx")

ops = {}

for node in model.graph.node:
    ops[node.op_type] = ops.get(node.op_type, 0) + 1
    
total = sum(p.numel() for p in model.parameters())
print(total)

print(ops)