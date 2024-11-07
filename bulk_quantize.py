import os
import subprocess
import shutil
import sys

# Directory paths
model_dir = "./Pruned_Model_Quantization_Compile/Float"
nndct_source = "./nndct"
quantized_dir = "./Pruned_Model_Quantization_Compile/Quantized"
compiled_src = "./compiled"
compiled_dir = "./Pruned_Model_Quantization_Compile/Compiled"
root_dir = "./"  # Root directory

# Ensure the quantized directory exists
os.makedirs(quantized_dir, exist_ok=True)

# Loop through all .pt files in the model directory
for model_file in os.listdir(model_dir):
    if model_file.endswith(".pt"):
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(model_dir, model_file)

        print("--" * 15)
        print("Starting Quantization of the ", model_name)
        print(f"Model path: {model_path}")  # Debugging statement

        # Create target directory for the quantized model
        target_dir = os.path.join(quantized_dir, model_name)
        os.makedirs(target_dir, exist_ok=True)

        # Target directory for compiled model
        target_compiled = os.path.join(compiled_dir, model_name)

        # Log file path
        log_file_path = os.path.join(quantized_dir, f"{model_name}.log")

        # Copy model to root directory as model.pt
        shutil.copy(model_path, os.path.join(root_dir, "model.pt"))

        # Prepare the commands
        commands = [
            f"python test_nndct.py --data data/voc.yaml --img 416 --batch 1 --conf 0.001 --iou 0.65 --device 'cpu' --weights model.pt --name yolov7_640_val --quant_mode calib --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish",
            f"python test_nndct.py --data data/voc.yaml --img 416 --batch 1 --conf 0.001 --iou 0.65 --device 'cpu' --weights model.pt --name yolov7_640_val --quant_mode test --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish",
            f"python test_nndct.py --data data/voc.yaml --img 416 --batch 1 --conf 0.001 --iou 0.65 --device 'cpu' --weights model.pt --name yolov7_640_val --quant_mode test --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish --dump_model",
            f"vai_c_xir -x nndct/Model_0_int.xmodel -a arch.json -o ./compiled -n {model_name}",
        ]

        # Open the log file
        with open(log_file_path, "w") as log:
            # Execute the commands
            for command in commands:
                print(f"Executing command: {command}")

                # Use Popen to get real-time output
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                # Read and process output in real-time
                for line in process.stdout:
                    # Write to log file
                    log.write(line)
                    log.flush()  # Ensure it's written immediately

                    # Print to screen
                    sys.stdout.write(line)
                    sys.stdout.flush()  # Ensure it's displayed immediately

                # Wait for the process to complete
                process.wait()

        # Copy the nndct folder to the quantized directory and rename it
        shutil.copytree(compiled_src, target_compiled)

        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        shutil.copytree(nndct_source, target_dir)

        print(f"Processed model: {model_name}")

print("All models processed.")
