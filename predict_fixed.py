"""
Fixed predict script using correct scaler (87 features)
This ensures scaler matches the enhanced models
"""
import sys
import subprocess

# Default to using scaler_enhanced_87.pkl which matches the enhanced models
args = sys.argv[1:]

# Check if --scaler is already specified
has_scaler = False
for i, arg in enumerate(args):
    if arg == '--scaler' and i + 1 < len(args):
        has_scaler = True
        break

# If no scaler specified, use the correct one
if not has_scaler:
    # Find position to insert scaler argument (after --model)
    try:
        model_idx = args.index('--model')
        # Insert scaler after model and its value
        args.insert(model_idx + 2, '--scaler')
        args.insert(model_idx + 3, 'models/scaler_enhanced_87.pkl')
    except ValueError:
        # No --model specified, append at end
        args.extend(['--scaler', 'models/scaler_enhanced_87.pkl'])

# Run the actual predict script
cmd = ['python', 'src/predict.py'] + args
result = subprocess.run(cmd)
sys.exit(result.returncode)
