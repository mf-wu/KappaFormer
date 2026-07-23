import os
import random
from ase.io import read
import lmdb
import pickle as pkl
import numpy as np
from tqdm import tqdm

def convert_to_lmdb(data_name,
                    split=None,
                    trajs=None,
                    save_path="/home/work/data/test",
                    random_translation=False,
                    max_atoms_num=10000,
                    ):
    max_atoms = 0
    nan_inf_count = 0

    if trajs is None:
        trajs = read(f"{data_path}/{data_name}.xyz", ":", format="extxyz")
    # 读pkl
    #trajs = []
    # with open(data_path, "rb") as f:
    #     trajs = pkl.load(f)
    if split is None:
        data_save_path = f"{save_path}/{data_name}"
    else:
        data_save_path = f"{save_path}/{data_name}/{split}"
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    write_env = lmdb.open(data_save_path, map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)
    for i, traj in tqdm(enumerate(trajs)):
        # traj = traj * (2,2,2)
        key = f"{data_name}-{i}"
        graph = traj.todict()
        graph["data_name"] = data_name
        trajs_atoms = len(traj)
        if trajs_atoms > max_atoms_num:
            continue
        if trajs_atoms > max_atoms:
            max_atoms = trajs_atoms
        if "info" not in graph:
            graph["info"] = {} 

        pos = traj.get_positions()
        graph["positions"] = pos
        # -------------------------------------------------------------

        write_txn.put(key.encode(), pkl.dumps(graph))
    write_txn.commit()
    
   
if __name__ == "__main__":
    data_path_dir = "/share/home/u15502/mfwu/kappaformer/data/kappaformer/htc_cal"
    save_path_dir = "/share/home/u15502/mfwu/kappaformer/data/kappaformer/htc_cal_lmdb"
    data_path_list = os.listdir(data_path_dir)
    # data_path_list = ["atoms_list_merged.pkl"]

    print(data_path_list)
    for data_path in data_path_list:
        data_path = os.path.join(data_path_dir, data_path)
        data_name = data_path.split("/")[-1].split(".")[0]

        print(f"processing {data_path}")
        if os.path.exists(f"{save_path_dir}/{data_name}"):
            print(f"{save_path_dir}/{data_name} exists, rewrite it")
            # continue

        if data_path.split(".")[-1] == "xyz":
            trajs = read(data_path, ":", format="extxyz")
        elif data_path.split(".")[-1] == "extxyz":
            trajs = read(data_path, ":", format="extxyz")
        else:
            with open(data_path, "rb") as f:
                trajs = pkl.load(f)

        random.seed(2000)
        random.shuffle(trajs)
        # train_trajs = trajs[:int(len(trajs) * 0.9)]
        # valid_trajs = trajs[int(len(trajs) * 0.9):int(len(trajs) * 0.95)]
        # test_trajs = trajs[int(len(trajs) * 0.95):]
        # print(f"train: {len(train_trajs)}, valid: {len(valid_trajs)}, test: {len(test_trajs)}")

        n = len(trajs)
        train_trajs = trajs[:int(n * 0.8)]
        valid_trajs = trajs[int(n * 0.8):int(n * 0.9)]
        test_trajs = trajs[int(n * 0.9):]

        convert_to_lmdb(data_name,
                        "train", 
                        train_trajs,
                        save_path=save_path_dir,
                        max_atoms_num=1000,
                        )
        convert_to_lmdb(data_name,
                        "valid", 
                        valid_trajs,
                        save_path=save_path_dir,
                        max_atoms_num=1000,
                        )
        convert_to_lmdb(data_name,
                        "test", 
                        test_trajs,
                        save_path=save_path_dir,
                        max_atoms_num=1000,
                        )