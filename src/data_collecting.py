# -*- coding: utf-8 -*-

import requests, zipfile, io, os, shutil


def dl_and_extract_zip(url, save_to):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(save_to)


def del_unwanted_files(directory, kept_files):
    dir_files = os.listdir(directory)
    removed_files = [f for f in dir_files if f not in kept_files]
    for f in removed_files:
        os.remove(f"{directory}/{f}")


def collect_data():
    # remove existing files from directory
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sec_dir = f"{app_dir}/data/raw/SEC"
    dirs = os.listdir(sec_dir)
    if len(dirs) > 0:
        for d in dirs:
            shutil.rmtree(f"{sec_dir}/{d}")
    # dowload SEC data
    sec_data = "https://www.sec.gov/foia/docs/adv/form-adv-complete-ria.zip"
    dl_and_extract_zip(sec_data, sec_dir)

    # remove files that aren't Base A questions
    zipped_loc = sec_dir + "/" + os.listdir(sec_dir)[0]
    files = os.listdir(zipped_loc)
    kept_files = [f for f in files if f.find("IA_ADV_Base_A") > -1]
    del_unwanted_files(zipped_loc, kept_files)


if __name__ == "__main__":

    collect_data()
