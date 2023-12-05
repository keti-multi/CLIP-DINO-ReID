import torch
model1_weights = torch.load('/data/keti/syh/checkpoints/CLIP_ReID_pure_Test_on_MSMT17_re/ViT-B-16_60.pth')
# model2_weights = torch.load('/data/keti/syh/checkpoints/CLIP_ReID_MSMT_baseline/ViT-B-16_60.pth')
model2_weights = torch.load('/data/keti/syh/checkpoints/CLIP_ReID_MSMT17/MSMT17_clipreid_ViT-B-16_60.pth')

for key in model2_weights:
    print(f"파라미터: {key}")
    if key in model1_weights:
        param1 = model1_weights[key]
        param2 = model2_weights[key]
        print(f"  Model 1 - 사이즈: {param1.size()}, 타입: {param1.dtype}")
        print(f"  Model 2 - 사이즈: {param2.size()}, 타입: {param2.dtype}")

        if param1.size() != param2.size() or param1.dtype != param2.dtype:
            print("  -> 파라미터 사이즈나 타입이 다릅니다.")
        else:
            print("  -> 파라미터 사이즈와 타입이 동일합니다.")
    else:
        print("  -> Model 2에 해당 파라미터가 존재하지 않습니다.")