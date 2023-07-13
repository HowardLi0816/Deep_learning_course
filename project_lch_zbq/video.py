import torch
from torchvision import transforms
import numpy as np
import cv2
import PIL
from model.CNN5 import CNN5
from utils.helpers import load_checkpoint


def preprocess(image):
    img_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((112,112)),
                                         transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2719, 0.2654, 0.2743])])
    image = img_transforms(image)
    image = image.float().unsqueeze(0)
    return image


def get_result(pred, labels):
    pred = pred.cpu().detach().numpy()
    prediction = np.argmax(pred, axis=1)
    result = labels[prediction[0]]
    return result


def main():
    labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
              6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
              12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
              18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
              24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}
    model = CNN5(flatten=7)
    # model_path = f"./CNN5_model/best_CNN5_pretrain_True_imgsize_112_bs_32.pth"
    model_path = f"./beifen/70/best_CNN5_pretrain_True_imgsize_112_bs_32.pth"
    model_state_dict, optimizer_state_dict = load_checkpoint(model_path)
    model.load_state_dict(model_state_dict)

    capture = cv2.VideoCapture(0)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    shape = (frame_width, frame_height)
    video_usc = cv2.VideoWriter('video_USC.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, shape)

    text = ""
    count = 0
    while capture.isOpened():
        with torch.no_grad():
            model.eval()
            _, frame = capture.read()
            if count == 60:
                count = 0
                image = frame.copy()
                image_select = image[210:410, 10:210]
                image_data = preprocess(image_select)
                pred = model(image_data)
                res = get_result(pred, labels)
                if res == "del":
                    text = text[:-1]
                elif res == "nothing":
                    text = text + ""
                elif res == "space":
                    text = text + " "
                else:
                    text = text + res

        cv2.putText(frame, '%s' %(text), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.rectangle(frame, (10, 210), (210, 410), (250, 0, 0), 2)
        video_usc.write(frame)
        cv2.imshow("window1", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        count = count + 1

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()