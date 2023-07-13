import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

def change(file_path, save_path):
    img = cv2.imread(file_path)
    resized_img = cv2.resize(img, (200, 200))
    cv2.imwrite(save_path, resized_img)

def removebg(file_path, save_path, ts):
    img = cv2.imread(file_path)
    smg = SelfiSegmentation()
    newimg = smg.removeBG(img, threshold=ts)
    cv2.imwrite(save_path, newimg)

if __name__ == '__main__':
    ab = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing',
          'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    n = 115
    for i in range(115, 116):
        file_path = f"../data/archive/asl_alphabet_test/our_test/tp ({i+1}).jpg"
        save_path = f"../data/imgd/{ab[i // 4]}_test_{(i%4)+2}.jpg"
        change(file_path, save_path)

    n = 116
    for i in range(n):
        file_path = f"../data/archive/asl_alphabet_test/asl_alphabet_test_or/{ab[i // 4]}/{ab[i // 4]}_test_{(i % 4) + 2}.jpg"
        save_path = f"../data/archive/asl_alphabet_test/asl_alphabet_test_2/{ab[i // 4]}/{ab[i // 4]}_test_{(i % 4) + 2}.jpg"
        removebg(file_path, save_path, 0.2)



