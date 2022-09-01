import os
import subprocess

input_dir = 'raw_inference_data'
output_dir = 'inference_data/HR_hazy'

def main():
    files = os.listdir(input_dir)
    if len(files): 
        input(f"Warning: {input_dir} is not empty, hit ENTER to continue\n")
        
    for file_name in files:
        image_name = file_name.split('.')[0]
        image_name_out = image_name + '_608x448'
        file_name_out = image_name_out + '.png'

        path_to_input_file = os.path.join(input_dir, file_name)
        path_to_output_file = os.path.join(output_dir, file_name_out)

        subprocess.run(['ffmpeg', '-y', '-i', path_to_input_file, '-vf', 'scale=608:448', path_to_output_file])

if __name__ == "__main__":
    main()