from aitemplate.frontend import Tensor


def mark_output(tensor: Tensor, name: str):
    tensor._attrs["is_output"] = True
    tensor._attrs["name"] = name
    shape = [d._attrs["values"] for d in tensor._attrs["shape"]]
    print(f"AIT output `{name}` shape {shape}")
    return tensor
