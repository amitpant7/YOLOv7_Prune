import torch
import yaml
from collections import OrderedDict
import os


def update_yaml_config(
    pruned_model_path, org_yaml="./cfg/training/yolov7.yaml", name="pruned_yolo"
):
    """
    Update the YAML configuration of a pruned model.

    This function loads a pruned model, extracts its convolutional layer channels, updates the YAML configuration
    file with these channels, and returns the updated model dictionary.

    Args:
        pruned_model_path (str): Path to the pruned model file (checkpoint).
        org_yaml (str): Path to the original YAML configuration file. Default is './cfg/training/yolov7.yaml'.

    Returns:
        dict: Updated model dictionary with the new YAML configuration.
    """

    def load_pruned_model(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if "model" in checkpoint:
            return checkpoint["model"]
        else:
            raise KeyError("The 'model' key was not found in the checkpoint file.")

    def trace_model_channels(model):
        channels = OrderedDict()
        conv_count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                channels[conv_count] = module.out_channels
                conv_count += 1
        return channels

    def update_yaml_channels(yaml_data, channels):
        conv_count = 0

        def update_layer_channels(layer):
            nonlocal conv_count
            if (
                isinstance(layer, list)
                and len(layer) == 4
                and layer[2] in ["Conv", "RepConv"]
            ):
                if conv_count in channels:
                    layer[3][0] = channels[conv_count]
                    conv_count += 1
            return layer

        # Update backbone and head
        for section in ["backbone", "head"]:
            yaml_data[section] = [
                update_layer_channels(layer) for layer in yaml_data[section]
            ]

        return yaml_data

    def custom_yaml_dump(data, stream=None, **kwargs):
        class OrderedDumper(yaml.SafeDumper):
            pass

        def _dict_representer(dumper, data):
            return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

        def represent_list_inline(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        def custom_str_presenter(dumper, data):
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        OrderedDumper.add_representer(list, represent_list_inline)
        OrderedDumper.add_representer(str, custom_str_presenter)

        return yaml.dump(data, stream, OrderedDumper, **kwargs)

    # Load pruned model and trace its channels
    model = load_pruned_model(pruned_model_path)
    channels = trace_model_channels(model)

    # Load and update original YAML configuration
    with open(org_yaml, "r") as file:
        original_yaml = yaml.safe_load(file)
    updated_yaml = update_yaml_channels(original_yaml, channels)

    output_yaml_path = f"{name}_cfg.yaml"
    with open(output_yaml_path, "w") as file:
        custom_yaml_dump(updated_yaml, file, default_flow_style=False, sort_keys=False)

    print(f"Updated YAML file saved to {output_yaml_path}")

    # Update the model dictionary with the new YAML configuration
    model_dict = torch.load(pruned_model_path)
    model_dict["model"].yaml = updated_yaml

    return model_dict


# Example usage
# updated_model = update_yaml_config('path_to_pruned_model.pth')
