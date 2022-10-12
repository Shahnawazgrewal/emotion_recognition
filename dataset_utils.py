import pickle

def read_file(file_name):
    feats = []
    labels = []
    with open(file_name, 'rb') as handle:
        objects = pickle.load(handle)
        for item in range(len(objects)):
            sfeat = objects[item][0][0]
            slabel = objects[item][1]
            slabel = slabel.split('_')[1:]
            if slabel == 'pleasant_surprise':
                slabel = 'pleasant_surprised'
            feats.append(sfeat)
            labels.append(slabel[0].lower())
    return feats, labels
