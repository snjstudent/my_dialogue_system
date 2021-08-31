from glob import glob
import os
import csv

base_jpg = 'fashionWOMENTees_Tanksid0000660217_4full.jpg'
csv_filename = 'intermediate/adgan_location_pair.csv'
npy_files = sorted(glob("intermediate/location_adgan/*"))
base_jpg_noex = base_jpg.split(".")[0]
csv_datas = [["from", "to"]]
if os.path.exists(csv_filename):
    os.system(f'rm -rf {csv_filename}')
for npy_file in npy_files:
    os.system(
        f'cp {npy_file} fashion_resize/testK/{base_jpg_noex}_{npy_file.split("/")[1].split(".")[0]}.jpg.npy')
    os.system(
        f'cp fashion_resize/test/{base_jpg} fashion_resize/test/{base_jpg_noex}_{npy_file.split("/")[1].split(".")[0]}.jpg')
    csv_datas.append(
        [base_jpg, f'{base_jpg_noex}_{npy_file.split("/")[1].split(".")[0]}.jpg'])
with open('intermediate/adgan_location_pair.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_datas)
    
