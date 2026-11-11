import cv2
import numpy as np
import os
import csv
from skimage.metrics import structural_similarity as ssim

def degrade_image(img, alpha=0.5, noise_sigma=15):
    """Riduce il contrasto e aggiunge rumore Gaussiano."""
    np.random.seed(42) 
    
    # Riduzione contrasto lineare
    degraded = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    
    # Aggiunta rumore Gaussiano
    noise = np.random.normal(0, noise_sigma, degraded.shape).astype(np.float32)
    degraded_noisy = cv2.add(degraded.astype(np.float32), noise)
    
    return np.clip(degraded_noisy, 0, 255).astype(np.uint8)

def linear_contrast_stretching(img):
    """Stretching sul canale L dello spazio LAB per evitare shift cromatici."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_stretched = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    lab_stretched = cv2.merge([l_stretched, a, b])
    return cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2BGR)

def histogram_equalization(img):
    """Equalizzazione sul canale V dello spazio HSV."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

def gamma_correction(img, gamma=1.5):
    """Quarto metodo: correzione Gamma per l'illuminazione non uniforme."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def compute_metrics(original, processed):
    """Calcola PSNR e SSIM."""
    psnr = cv2.PSNR(original, processed)
    ssim_val = ssim(original, processed, channel_axis=2) 
    return psnr, ssim_val

def main():
    os.makedirs("results", exist_ok=True)
    images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"] 
    results_csv = [["Image", "Method", "PSNR", "SSIM"]]
    
    for img_name in images:
        img_path = img_name
        original = cv2.imread(img_path)
        if original is None:
            continue
            
        degraded = degrade_image(original)
        cv2.imwrite(f"results/degraded_{img_name}", degraded)
        
        
        methods = {
            "Baseline": degraded,
            "Linear_Stretching": linear_contrast_stretching(degraded),
            "Hist_Equalization": histogram_equalization(degraded),
            "Gamma_Correction": gamma_correction(degraded)
        }
        
        for method_name, result in methods.items():
            cv2.imwrite(f"results/{method_name}_{img_name}", result)
            psnr, ssim_val = compute_metrics(original, result)
            results_csv.append([img_name, method_name, round(psnr, 2), round(ssim_val, 4)])
            
    
    with open("results/metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results_csv)
    print("Elaborazione completata. Risultati in /results.")

if __name__ == "__main__":
    main()