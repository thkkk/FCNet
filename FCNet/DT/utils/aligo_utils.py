from typing import List
import os
from os.path import exists, join
import shutil
import aligo
from aligo import Aligo
from copy import deepcopy

def lis_dir(ali: Aligo, dir_id):
    '''
    get file list 传入默认是'root' 根目录
    file_list, file_name_list, file_id_list, file_type_list = lis_dir(ali, dir_path[i])
    '''
    ll = ali.get_file_list(dir_id)
    return ll, list(map(lambda x: x.name, ll)), list(map(lambda x: x.file_id, ll)), list(map(lambda x: x.type, ll))

def print_user_info(ali: Aligo):
    user = ali.get_user()
    print(user.user_name, user.nick_name, user.phone)

def mkdir(ali: Aligo, dir_id, dir_name: str):
    os.makedirs(dir_name)
    ali.upload_folder(dir_name, parent_file_id=dir_id)
    shutil.rmtree(dir_name)

def mkdirp(ali: Aligo, dir_path: List[str]):
    '''
    @param dir_path: start with 'root'
    '''
    dir_path = deepcopy(dir_path)
    dir_path[0] = ali.get_folder_by_path(dir_path[0]).file_id
    for i in range(len(dir_path) - 1):
        file_list, file_name_list, file_id_list, file_type_list = lis_dir(ali, dir_path[i])
        if dir_path[i+1] not in file_name_list:
            mkdir(ali, dir_path[i], dir_path[i+1])
            file_list, file_name_list, file_id_list, file_type_list = lis_dir(ali, dir_path[i])
        dir_path[i+1] = file_id_list[file_name_list.index(dir_path[i+1])]

def upload_folder_no_duplicate(ali: Aligo, parent_folder_id, file_name_list: list, local_dir_path: str):
    local_dir_name = local_dir_path.split('/')[-1]
    if local_dir_name not in file_name_list:
        ali.upload_folder(local_dir_path, parent_file_id=parent_folder_id)

def download_folder_to_local(ali: Aligo, aligo_folder_path: str, local_parent_dir: str):
    aligo_folder_file = ali.get_folder_by_path(aligo_folder_path)
    assert aligo_folder_file is not None, f"aligo {aligo_folder_path} not found"
    os.makedirs(local_parent_dir, exist_ok=True)
    ali.download_folder(aligo_folder_file.file_id, local_parent_dir)