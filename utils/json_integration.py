import json
import sys

def process_json_input(json_path, model_path, output_path):
    with open(json_path, 'r') as f:
        req = json.load(f)
    access_token = req.get('access_token')
    allowed = access_token in [
        'rudra_qazwsxedcrfvtgbyhnujmikolp',
        'rudra_plokmijnuhbygvtfcrdxeszwaq'
    ]
    if not allowed:
        raise ValueError('Unauthorized access_token')
    prompt = req['prompt']
    duration = int(req['duration'])
    # Call aura.cpp or model for generation
    # Placeholder: print parameters
    print(f"Authorized: Generating '{prompt}' for {duration} seconds -> {output_path}")
    # (real call: pass to C++/Python model runner)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <json_file> <model_path> <output_path>")
        exit(1)
    process_json_input(sys.argv[1], sys.argv[2], sys.argv[3])
