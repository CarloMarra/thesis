import time

import torch
from flags import FLAGS
from resnet import SlimmableResNet50


def count_active_params(model, width_mult):
    """Count parameters actively used at a given width."""
    model.switch_to_width(width_mult)
    
    active_params = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'current_in_channels'):
            # Count active parameters in conv layers
            active_weight_params = module.current_out_channels * module.current_in_channels * module.kernel_size[0] * module.kernel_size[1]
            if module.bias is not None:
                active_weight_params += module.current_out_channels
                
            # Total parameters in this layer
            total_weight_params = module.weight.numel()
            if module.bias is not None:
                total_weight_params += module.bias.numel()
                
            active_params += active_weight_params
            total_params += total_weight_params
            
            # print(f"{name}: Using {active_weight_params:,}/{total_weight_params:,} parameters ({active_weight_params/total_weight_params:.1%})")
    
    return active_params, total_params

def measure_inference_time(model, width_mult, batch_size=32, num_runs=50):
    """Measure actual inference time for a given width."""
    model.switch_to_width(width_mult)
    model.eval()
    
    # Create input with specified batch size
    x = torch.randn(batch_size, 3, 224, 224, device='mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def test_slimmable_resnet():
    print("Creating SlimmableResNet50...")
    model = SlimmableResNet50()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    
    # Set a larger batch size
    batch_size = 32
    
    results = {}
    
    for width in FLAGS.width_mult_list:
        print(f"\n===== Testing width multiplier: {width} =====")
        model.switch_to_width(width)
        
        # Test forward pass with larger batch
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        try:
            with torch.no_grad():
                outputs = model(x)
            
            if isinstance(outputs, tuple):
                print(f"Output feature maps: {len(outputs)}")
                for i, feat in enumerate(outputs):
                    # Check for actual non-zero values
                    active_channels = (feat.abs().sum([0,2,3]) > 0).sum().item()
                    print(f"  Feature map {i}: {feat.shape}, Active channels: {active_channels}/{feat.shape[1]}")
            else:
                print(f"Output shape: {outputs.shape}")
            
            # Count active parameters
            active_params, total_params = count_active_params(model, width)
            
            # Measure inference time
            batch_size = 32  # Same as above
            inference_time = measure_inference_time(model, width, batch_size=batch_size)
            
            # Calculate per-sample time for fair comparison
            time_per_sample = inference_time / batch_size
            
            results[width] = {
                'active_params': active_params,
                'total_params': total_params,
                'param_efficiency': active_params / total_params,
                'inference_time': inference_time,
                'time_per_sample': time_per_sample
            }
            
            print(f"Active parameters: {active_params:,}/{total_params:,} ({active_params/total_params:.2%})")
            print(f"Inference time: {inference_time*1000:.2f} ms (batch), {time_per_sample*1000:.2f} ms per sample")
            
        except Exception as e:
            print(f"Error at width {width}: {str(e)}")
    
    # Compare results
    print("\n===== Summary =====")
    baseline_width = max(FLAGS.width_mult_list)
    baseline_time = results[baseline_width]['inference_time']
    baseline_params = results[baseline_width]['active_params']
    
    print(f"Width | Param % | Speedup | Time (ms)")
    print(f"------+---------+---------+----------")
    
    for width in sorted(FLAGS.width_mult_list):
        param_ratio = results[width]['active_params'] / baseline_params
        time_ratio = baseline_time / results[width]['inference_time']
        time_ms = results[width]['inference_time'] * 1000
        
        print(f"{width:.2f}  | {param_ratio:.2%}  | {time_ratio:.2f}x   | {time_ms:.2f}")

if __name__ == "__main__":
    test_slimmable_resnet()