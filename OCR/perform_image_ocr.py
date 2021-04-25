import easyocr
import glob
import pickle


def perform_recognition():
    reader = easyocr.Reader(['en'], gpu=True)
    res_dict = dict()
    for im in glob.glob('./data/*.jpg'):
        encoded = reader.readtext(im, detail=0)
        res_dict[im.split('.')[1]] = encoded
    return res_dict


if __name__ == '__main__':
    dct = perform_recognition()
    with open('./data/recognized_dict.pickle', 'wb') as f:
        pickle.dump(dct, f)
