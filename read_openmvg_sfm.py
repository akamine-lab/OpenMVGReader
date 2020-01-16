
import json
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("json_path")
parser.add_argument("image_dir")
parser.add_argument("object_pos_x",type=float)
parser.add_argument("object_pos_y",type=float)
parser.add_argument("object_pos_z",type=float)
args = parser.parse_args()

OBJ_POS = [args.object_pos_x,args.object_pos_y,args.object_pos_z]
print("object pos = ", OBJ_POS)

json_file = args.json_path
image_dir = args.image_dir

class View:
    def __init__(self, key, image, K,Rt):
        self.key = 0
        self.image = image
        self.intrinsic = K
        self.extrinsic = Rt

class SfmDataReader:
    def __init__(self, sfm_data_path):
        with open(json_file) as f:
            sfm_data = json.load(f)

        self.json = sfm_data
        self.exrinsics = {}
        self.intrinsics = {}
        self.views = {}

    def parse(self):
        intrs = self.json["intrinsics"]
        extrs = self.json["extrinsics"]
        views = self.json["views"]

        for intr in intrs:
            v = intr["value"]["ptr_wrapper"]["data"]
            f = v["focal_length"]
            cx = v["principal_point"][0]
            cy = v["principal_point"][1]
            self.intrinsics[intr["key"]] = self.getIntrinsic(f,cx,cy)

        for ex in extrs:
            j = ex["value"]
            r = np.array(j["rotation"])
            t = np.array(j["center"])

            ext = np.identity(4)
            ext[0:3,0:3] = r.T
            ext[0:3,3:4] = t[:,None]
            ext = np.linalg.inv(ext)

            self.exrinsics[ex["key"]] = ext[0:3,:]

        for view in views:
            key = view["key"]
            data = view["value"]["ptr_wrapper"]["data"]
            file = data["filename"]
            K = self.intrinsics[data["id_intrinsic"]]
            Rt = self.exrinsics[data["id_pose"]]
            self.views[key] = View(key, file, K, Rt)

    def getIntrinsic(self,f,cx,cy):
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]])
        return K

def project(p, K, Rt):
    proj = K @ Rt @ obj_pos
    proj = proj.flatten()
    proj /= proj[2]
    pos =  proj[0:2].astype(np.int).tolist()
    return (pos[1],pos[0])

reader = SfmDataReader(json_file)
reader.parse()

OBJ_POS.append(1)
obj_pos = np.array(OBJ_POS)[:,None]

for view in reader.views.values():
    image = cv2.imread(image_dir+"/"+view.image)
    pos = project(obj_pos, view.intrinsic, view.extrinsic)
    pos = (image.shape[1]-pos[0], pos[1])
    image = cv2.circle(image, pos, 20, (255,255,0),5)

    print(pos)
    #縮小して表示
    image = cv2.resize(image, None, None, 0.25, 0.25)
    cv2.imshow("projection", image)
    cv2.waitKey()
