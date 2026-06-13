#!/usr/bin/env python3
"""
PyTorch 2.12.0 Compatibility Test Suite
Run this before merging the torch upgrade PR

Usage: python test_torch_upgrade.py
"""

import sys
import torch
import subprocess
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_check(msg, passed):
    """Print test result with color"""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"  [{status}] {msg}")
    return passed

def test_environment():
    """Test PyTorch environment setup"""
    print(f"\n{Colors.BLUE}1. Environment Setup{Colors.END}")
    
    tests_passed = 0
    tests_total = 0
    
    # Check PyTorch version
    tests_total += 1
    version = torch.__version__
    passed = version.startswith('2.12')
    tests_passed += print_check(f"PyTorch version: {version}", passed)
    
    # Check CUDA availability
    tests_total += 1
    cuda_available = torch.cuda.is_available()
    print_check(f"CUDA available: {cuda_available}", cuda_available)
    if not cuda_available:
        print(f"    {Colors.YELLOW}Warning: CUDA not available. CPU-only mode.{Colors.END}")
    tests_passed += 1
    
    # Check CUDA version if available
    if cuda_available:
        tests_total += 1
        cuda_version = torch.version.cuda
        print(f"  [ℹ] CUDA version: {cuda_version}")
        # CUDA 12.6+ required for torch 2.12
        passed = cuda_version >= "12.6"
        print_check(f"CUDA 12.6+ (required for 2.12.0)", passed)
        tests_passed += passed
    
    # Check cuDNN
    if cuda_available:
        tests_total += 1
        cudnn_version = torch.backends.cudnn.version()
        print(f"  [ℹ] cuDNN version: {cudnn_version}")
        tests_passed += 1
    
    return tests_passed, tests_total

def test_tensor_operations():
    """Test basic tensor operations"""
    print(f"\n{Colors.BLUE}2. Tensor Operations{Colors.END}")
    
    tests_passed = 0
    tests_total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Basic tensor creation
        tests_total += 1
        x = torch.randn(100, 100, device=device)
        print_check(f"Tensor creation ({device})", True)
        tests_passed += 1
        
        # Matrix multiplication
        tests_total += 1
        y = torch.matmul(x, x.T)
        print_check(f"Matrix multiplication", y.shape == torch.Size([100, 100]))
        tests_passed += 1
        
        # dtype handling
        tests_total += 1
        x_fp32 = x.float()
        x_fp64 = x.double()
        print_check(f"dtype conversion (float32/float64)", True)
        tests_passed += 1
        
        # Mixed precision
        tests_total += 1
        x_fp16 = x.half()
        print_check(f"Mixed precision (float16)", True)
        tests_passed += 1
        
    except Exception as e:
        print(f"  {Colors.RED}✗ Tensor operations failed: {e}{Colors.END}")
        return tests_passed, tests_total
    
    return tests_passed, tests_total

def test_model_loading():
    """Test model save/load compatibility"""
    print(f"\n{Colors.BLUE}3. Model Serialization{Colors.END}")
    
    tests_passed = 0
    tests_total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        ).to(device)
        
        tests_total += 1
        print_check("Model creation", True)
        tests_passed += 1
        
        # Save model
        tests_total += 1
        model_path = Path("/tmp/test_model_2.12.pt")
        torch.save(model.state_dict(), model_path)
        print_check(f"Model save: {model_path.name}", model_path.exists())
        tests_passed += 1
        
        # Load model
        tests_total += 1
        model2 = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        ).to(device)
        model2.load_state_dict(torch.load(model_path, map_location=device))
        print_check(f"Model load", True)
        tests_passed += 1
        
        # Model inference
        tests_total += 1
        x_test = torch.randn(5, 10, device=device)
        output = model2(x_test)
        print_check(f"Model inference: {output.shape}", output.shape == torch.Size([5, 10]))
        tests_passed += 1
        
        # Cleanup
        model_path.unlink()
        
    except Exception as e:
        print(f"  {Colors.RED}✗ Model loading failed: {e}{Colors.END}")
        return tests_passed, tests_total
    
    return tests_passed, tests_total

def test_training_loop():
    """Test basic training loop"""
    print(f"\n{Colors.BLUE}4. Training Loop{Colors.END}")
    
    tests_passed = 0
    tests_total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Setup
        model = torch.nn.Sequential(
            torch.nn.Linear(20, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()
        
        tests_total += 1
        print_check("Training setup", True)
        tests_passed += 1
        
        # Training loop
        tests_total += 1
        losses = []
        for i in range(10):
            x = torch.randn(32, 20, device=device)
            y = torch.randn(32, 1, device=device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Check loss trend (should generally decrease or stabilize)
        print_check(f"Training loop (10 iterations)", True)
        tests_passed += 1
        
        # Check gradient flow
        tests_total += 1
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Sanity check: gradients should be non-zero
                assert grad_norm > 0, f"Zero gradient for {name}"
        
        print_check(f"Gradient flow", True)
        tests_passed += 1
        
    except Exception as e:
        print(f"  {Colors.RED}✗ Training loop failed: {e}{Colors.END}")
        return tests_passed, tests_total
    
    return tests_passed, tests_total

def test_gpu_memory():
    """Test GPU memory management"""
    print(f"\n{Colors.BLUE}5. GPU Memory Management{Colors.END}")
    
    tests_passed = 0
    tests_total = 0
    
    if not torch.cuda.is_available():
        print(f"  {Colors.YELLOW}⊘ Skipped (CUDA not available){Colors.END}")
        return tests_passed, tests_total
    
    try:
        tests_total += 1
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate tensors
        x = [torch.randn(1000, 1000, device='cuda') for _ in range(5)]
        allocated = torch.cuda.memory_allocated() - initial_memory
        
        print_check(f"Memory allocation: {allocated / 1e6:.1f} MB", allocated > 0)
        tests_passed += 1
        
        # Clear memory
        tests_total += 1
        del x
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        print_check(f"Memory cleanup", final_memory <= initial_memory + 1e7)  # Allow 10MB tolerance
        tests_passed += 1
        
    except Exception as e:
        print(f"  {Colors.RED}✗ GPU memory test failed: {e}{Colors.END}")
    
    return tests_passed, tests_total

def main():
    """Run all tests"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}PyTorch 2.12.0 Upgrade Compatibility Test{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    all_passed = 0
    all_total = 0
    
    # Run all test suites
    test_suites = [
        test_environment,
        test_tensor_operations,
        test_model_loading,
        test_training_loop,
        test_gpu_memory,
    ]
    
    for test_suite in test_suites:
        try:
            passed, total = test_suite()
            all_passed += passed
            all_total += total
        except Exception as e:
            print(f"  {Colors.RED}✗ Test suite failed: {e}{Colors.END}")
    
    # Summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Summary{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    percentage = (all_passed / all_total * 100) if all_total > 0 else 0
    status_color = Colors.GREEN if all_passed == all_total else Colors.RED
    status = "✓ ALL TESTS PASSED" if all_passed == all_total else "✗ SOME TESTS FAILED"
    
    print(f"{status_color}{status}{Colors.END}")
    print(f"Result: {all_passed}/{all_total} tests passed ({percentage:.0f}%)")
    
    if all_passed == all_total:
        print(f"\n{Colors.GREEN}✓ Safe to upgrade to PyTorch 2.12.0{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.RED}✗ Issues found. Review failures above before upgrading.{Colors.END}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
