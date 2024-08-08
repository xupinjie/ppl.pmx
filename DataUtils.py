import torch

# for sd model
class __TensorDumperV2__:
    call_idx = 0
    def __init__(self, dir="."):
        self.enable_dump = True
        self.dir = dir
        self.step = 0
        self.dump_steps = []


    def dump(self, X: torch.Tensor, name: str, idx=0):
        if not self.enable_dump:
            return
        
        if len(self.dump_steps) > 0 and self.step not in self.dump_steps:
            return

        shape_str = "" if X.dim == 0 else str(X.shape[0])
        for d in X.cpu().numpy().shape[1:]:
            shape_str = shape_str + "_" + str(d)

        type_dict = {
            torch.float: "fp32",
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.int8: "int8",
            torch.int64: "int64",
        }

        name = name.replace("/", ".")
        filename = "{}_{}_output_{}-{}-{}.bin".format(__TensorDumperV2__.call_idx, name, idx, shape_str, type_dict[X.dtype])
        __TensorDumperV2__.call_idx+=1

        X.cpu().numpy().tofile(self.dir + "/" + filename)

# for llm model
class __TensorDumperV3__:
    call_idx = 0
    def __init__(self):
        self.enable_dump = True
        self.dir = dir
        self.step = 0
        self.dump_steps = []


    def dump(self, X: torch.Tensor, name: str, idx=0):
        if not self.enable_dump:
            return
        
        if len(self.dump_steps) > 0 and self.step not in self.dump_steps:
            return

        shape_str = "" if X.dim == 0 else str(X.shape[0])
        for d in X.cpu().numpy().shape[1:]:
            shape_str = shape_str + "_" + str(d)

        type_dict = {
            torch.float: "fp32",
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.int8: "int8",
            torch.int64: "int64",
        }

        name = name.replace("/", ".")
        filename = "{}_step{}_{}_output_{}-{}-{}.bin".format(__TensorDumperV2__.call_idx, self.step, name, idx, shape_str, type_dict[X.dtype])
        __TensorDumperV2__.call_idx+=1

        X.cpu().numpy().tofile(self.dir + "/" + filename)

def getShape(X):
    shape_str = "" if X.dim == 0 else str(X.shape[0])
    for d in X.cpu().numpy().shape[1:]:
        shape_str = shape_str + "_" + str(d)

    type_dict = {
        torch.float: "fp32",
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.int8: "int8",
        torch.int64: "int64",
    }

    return shape_str

def getType(X):
    shape_str = "" if X.dim == 0 else str(X.shape[0])
    for d in X.cpu().numpy().shape[1:]:
        shape_str = shape_str + "_" + str(d)

    type_dict = {
        torch.float: "fp32",
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.int8: "int8",
        torch.int64: "int64",
    }

    return type_dict[X.dtype]
