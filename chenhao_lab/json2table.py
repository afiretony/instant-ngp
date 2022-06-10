import json

if __name__=="__main__":

    with open("output/test_config_4/results.json") as f:
        log = json.load(f)
    
    checkpoints = list(log['psnr_train'].keys())
    for c in checkpoints:
        print("| {:04d} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(int(c), log['time'][c], log['psnr_train'][c], log['psnr_test'][c], log['ssim_train'][c], log['ssim_test'][c]))
        