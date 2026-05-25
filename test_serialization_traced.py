import torch
from encodec import EncodecModel

# Load a pretrained model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)

# Test serialization
print("Testing serialization...")

# Option 1: Save/load state dict (RECOMMENDED - most portable)
torch.save(model.state_dict(), 'encodec_test.pth')
print("✓ Saved state dict")

loaded_model = EncodecModel.encodec_model_24khz(pretrained=False)
loaded_model.load_state_dict(torch.load('encodec_test.pth', weights_only=True))
print("✓ Loaded state dict")

# Option 2: Save/load entire model (need weights_only=False)
torch.save(model, 'encodec_full.pth')
print("✓ Saved full model")

loaded_full = torch.load('encodec_full.pth', weights_only=False)
print("✓ Loaded full model")

# Option 3: Try JIT tracing instead of scripting (more compatible)
try:
    dummy_input = torch.randn(1, 1, 24000)  # 1 second of audio
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save('encodec_traced.pt')
    print("✓ JIT traced and saved")

    loaded_traced = torch.jit.load('encodec_traced.pt')
    print("✓ Loaded traced model")
except Exception as e:
    print(f"✗ JIT tracing failed: {e}")

# Verify they work
print("\nVerifying loaded models...")
dummy_audio = torch.randn(1, 1, 24000)

try:
    output1 = loaded_model(dummy_audio)
    print(f"✓ State dict model works - output shape: {output1.shape}")
except Exception as e:
    print(f"✗ State dict model failed: {e}")

try:
    output2 = loaded_full(dummy_audio)
    print(f"✓ Full model works - output shape: {output2.shape}")
except Exception as e:
    print(f"✗ Full model failed: {e}")

try:
    output3 = loaded_traced(dummy_audio)
    print(f"✓ Traced model works - output shape: {output3.shape}")
except Exception as e:
    print(f"✗ Traced model failed: {e}")

print("\nSerialization tests complete!")