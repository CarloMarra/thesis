import time

import torch
from flags import FLAGS
from resnet_cl import SlimmableResNet34


def count_active_params(model, width_mult):
    """Count the number of parameters actively used at a given width."""
    model.switch_to_width(width_mult)

    active_params = 0
    total_params = 0

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'current_in_channels'):
            w_active = (module.current_out_channels * 
                        module.current_in_channels * 
                        module.kernel_size[0] * module.kernel_size[1])
            if module.bias is not None:
                w_active += module.current_out_channels

            w_total = module.weight.numel()
            if module.bias is not None:
                w_total += module.bias.numel()

            active_params += w_active
            total_params += w_total

        elif isinstance(module, torch.nn.Linear) and hasattr(module, 'in_features_list'):
            idx = FLAGS.width_mult_list.index(model.width_mult)
            in_features = module.in_features_list[idx]
            out_features = module.out_features_list[idx]
            
            w_active = in_features * out_features
            if module.bias is not None:
                w_active += out_features

            w_total = module.weight.numel()
            if module.bias is not None:
                w_total += module.bias.numel()

            active_params += w_active
            total_params += w_total

    return active_params, total_params

def measure_inference_time(model, width_mult, batch_size=32, num_warmup=10, num_runs=50):
    """Measure average inference time for a given width and batch size."""
    model.switch_to_width(width_mult)
    model.eval()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    # Actual timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_runs
    return avg_time

def test_slimmable_resnet_classification():
    """Test the SlimmableResNet classification model at multiple width multipliers."""
    print("Initializing SlimmableResNet34 for classification...")
    batch_size = 16 
    num_classes = 1000 
    model = SlimmableResNet34(num_classes=num_classes)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    print(f"Using batch size: {batch_size}")

    # Summarize results
    results = {}

    for width in FLAGS.width_mult_list:
        print(f"\n=== Testing width multiplier: {width} ===")
        model.switch_to_width(width)

        # Create test input
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        print(f"Input shape: {x.shape}")

        with torch.no_grad():
            output = model(x)

        # Check output shape
        assert output.shape == (batch_size, num_classes), \
            f"Unexpected output shape: {output.shape}, expected ({batch_size}, {num_classes})"

        print(f"Output shape: {output.shape}")

        # Count active vs total parameters
        active_params, total_params = count_active_params(model, width)

        # Time a forward pass (batch-based)
        avg_time = measure_inference_time(model, width, batch_size=batch_size)
        time_per_sample = avg_time / batch_size

        results[width] = {
            'active_params': active_params,
            'total_params': total_params,
            'param_ratio': active_params / total_params,
            'avg_time_batch': avg_time,
            'time_per_sample': time_per_sample
        }

        print(f"Active params: {active_params:,}/{total_params:,} "
              f"({100 * active_params / total_params:.2f}%)")
        print(f"Avg. time per batch: {1000 * avg_time:.2f} ms, "
              f"per sample: {1000 * time_per_sample:.2f} ms")

    # Quick comparison summary
    print("\n=== Summary of results ===")
    baseline_width = max(FLAGS.width_mult_list)
    baseline_info = results[baseline_width]

    print(f"{'Width':>5} | {'Param%':>8} | {'Speedup':>7} | {'Time(ms/batch)':>15}")
    print("-" * 45)
    for w in sorted(FLAGS.width_mult_list):
        info = results[w]
        param_ratio = info['active_params'] / baseline_info['active_params']
        speedup = baseline_info['avg_time_batch'] / info['avg_time_batch']
        print(f"{w:>5.2f} | {param_ratio:>7.2%} |  {speedup:>6.2f}x | {info['avg_time_batch'] * 1000:>15.2f}")

if __name__ == "__main__":
    test_slimmable_resnet_classification()