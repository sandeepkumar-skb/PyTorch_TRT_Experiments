import torch
import torchvision
import argparse



def load_model_weight(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    return model

def export_onnx_model(model, dummy_input, onnx_path, input_names=None, output_names=None, dynamic_axes=None):
    model(dummy_input)
    torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, verbose=True)

def create_model(model_name):
    model = getattr(torchvision.models, model_name)
    return model(pretrained=True).cuda()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ONNX model")
    parser.add_argument("--model", \
            default="resnet18", \
            choices= ["resnet18", "resnet50", "vgg16", "vgg19"], \
            help='Choose a model', required=True)
    parser.add_argument("--model_path", \
            help="provide the model weight checkpoint", required=True)
    args = parser.parse_args()

    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    model = create_model(args.model)#torchvision.models.resnet18(pretrained=True).cuda()
    model_path = args.model_path #"checkpoints/resnet18-5c106cde.pth"
    model = load_model_weight(model, model_path)
    onnx_path = f"{args.model}.onnx"
    export_onnx_model(model, dummy_input, onnx_path)
