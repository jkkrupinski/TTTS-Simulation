import torch
import cv2
import numpy as np

from misc.TTTSNet import TTTSNet


class Model:
    def __init__(self, models, device):
        self.models = models
        self.device = device

    def __call__(self, x):
        preds = []
        x = x.to(self.device)

        with torch.no_grad():
            for m in self.models:
                pred = m(x)
                preds.append(pred)
        preds = torch.stack(preds)
        preds = torch.mean(preds, dim=0)
        return preds


class Segmenter:
    def __init__(self, model_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = []

        for info in model_list:
            m = TTTSNet(classes=2, num_features=64)
            m = torch.nn.DataParallel(m).to(device)
            checkpoint = torch.load(
                f"models/{info}", map_location="cpu", weights_only=True
            )

            new_checkpoint = self.rename_keys(checkpoint)

            m.load_state_dict(new_checkpoint)
            m.to(device)
            m.eval()
            models.append(m)

            self.model = Model(models, device)

    def rename_keys(self, checkpoint):
        new_checkpoint = {}
        for key in checkpoint.keys():
            new_key = key.replace("FFM_", "RFFM_")
            newer_key = new_key.replace("PMCA", "MEDCAM")

            new_checkpoint[newer_key] = checkpoint[key]

        return new_checkpoint

    def to_tensor(self, pic):
        if isinstance(pic, np.ndarray):
            # Handle numpy array (H x W x C) -> (C x H x W)
            img = (
                torch.from_numpy(pic).permute(2, 0, 1).float()
            )  # Convert to float and rearrange dimensions
            return img.div(255)  # Scale pixel values to [0, 1]

        elif pic.mode == "I":
            # Convert 32-bit integer images
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            # Convert 16-bit integer images
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            # Convert any other mode (PIL Image)
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # If the image has an alpha channel (e.g., RGBA)
        if pic.mode == "LA" or (pic.mode == "RGBA" and img.size(0) == 4):
            pic = pic.convert("RGB")  # Drop the alpha channel for compatibility

        # Convert image to numpy (H x W x C), if not already in this format
        nimg = np.array(pic, np.float32, copy=False)

        # Convert to torch tensor (C x H x W)
        img_tensor = torch.from_numpy(nimg).permute(2, 0, 1)

        return img_tensor.div(255.0)  # Scale to [0, 1]

    def normalize(self, tensor, mean, std):
        mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)

        return (tensor - mean) / std

    def process_input(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (448, 448))
        img = self.to_tensor(img)
        img = self.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = img.unsqueeze(0)
        return img

    def process_output(self, output):
        output = output.detach().squeeze().cpu().numpy()
        output = np.moveaxis(output, 0, -1)
        pred_mask = np.argmax(output, axis=2).astype("float32")
        pred_mask = cv2.resize(pred_mask, (256, 256))
        pred_mask = np.uint8(pred_mask)

        pred_mask *= 255
        return pred_mask

    def __call__(self, img):
        tensor = self.process_input(img)
        output = self.model(tensor)
        prediction = self.process_output(output)

        return prediction


if __name__ == "__main__":
    seg = Segmenter(
        [
            "TTTSNet_model-fold-0.pt",
            "TTTSNet_model-fold-1.pt",
            "TTTSNet_model-fold-2.pt",
            "TTTSNet_model-fold-3.pt",
            "TTTSNet_model-fold-4.pt",
            "TTTSNet_model-fold-5.pt",
        ]
    )
    pth = "dataset/images/test2.png"
    img = cv2.imread("a.png")
    out = seg(img)
    cv2.imshow("a", out)
    cv2.waitKey(0)
