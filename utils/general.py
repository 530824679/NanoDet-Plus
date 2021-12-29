

def xyxy2xywh(bbox):
    """
    change bbox to txt format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

def