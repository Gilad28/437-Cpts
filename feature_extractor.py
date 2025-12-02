import numpy as np
from skimage.feature import hog
from skimage import color
import cv2
from tqdm import tqdm

# Feature extraction - HOG works way better than color histograms btw
class FeatureExtractor:
    def __init__(self, feature_type='hog'):
        self.feature_type = feature_type
    
    def extract_hog(self, image):
        # convert to grayscale first
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image
        
        # tried different params, these work best
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys',
                      visualize=False, feature_vector=True)
        
        return features
    
    def extract_color_histogram(self, image, bins=32):
        # need to convert back to 0-255 for histograms
        img_uint8 = (image * 255).astype(np.uint8)
        
        histograms = []
        for i in range(3):  # R, G, B
            hist, _ = np.histogram(img_uint8[:, :, i], bins=bins, range=(0, 256))
            hist = hist.astype(np.float32) / (hist.sum() + 1e-7)  # normalize
            histograms.append(hist)
        
        return np.concatenate(histograms)
    
    def extract_features(self, image):
        if self.feature_type == 'hog':
            return self.extract_hog(image)
        elif self.feature_type == 'color':
            return self.extract_color_histogram(image)
        elif self.feature_type == 'combined':
            # combine both feature types
            hog_feat = self.extract_hog(image)
            color_feat = self.extract_color_histogram(image)
            return np.concatenate([hog_feat, color_feat])
        else:
            raise ValueError(f"idk what {self.feature_type} is")
    
    def extract_batch(self, images):
        print(f"\nExtracting {self.feature_type} features from {len(images)} images...")
        
        features = []
        for img in tqdm(images, desc="Extracting features"):
            feat = self.extract_features(img)
            features.append(feat)
        
        features = np.array(features, dtype=np.float32)
        print(f"Feature shape: {features.shape}")
        return features


def save_features(save_dir, X_train, y_train, X_val, y_val, X_test, y_test, 
                 label_names=None):
    # save all the feature arrays so we don't have to recompute
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving features to {save_dir}...")
    
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    
    if label_names is not None:
        np.save(os.path.join(save_dir, 'label_names.npy'), label_names)
    
    print("Done!")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")


def load_features(save_dir):
    # loads all the saved npy files
    import os
    
    print(f"\nLoading features from {save_dir}...")
    
    data = {
        'X_train': np.load(os.path.join(save_dir, 'X_train.npy')),
        'y_train': np.load(os.path.join(save_dir, 'y_train.npy')),
        'X_val': np.load(os.path.join(save_dir, 'X_val.npy')),
        'y_val': np.load(os.path.join(save_dir, 'y_val.npy')),
        'X_test': np.load(os.path.join(save_dir, 'X_test.npy')),
        'y_test': np.load(os.path.join(save_dir, 'y_test.npy')),
    }
    
    # load label names if they exist
    label_names_path = os.path.join(save_dir, 'label_names.npy')
    if os.path.exists(label_names_path):
        data['label_names'] = np.load(label_names_path, allow_pickle=True)
    
    print("Loaded!")
    return data

