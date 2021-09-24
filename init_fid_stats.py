import os
import glob
import shutil
import cleanfid

mod_path = os.path.dirname(cleanfid.__file__)
stats_folder = os.path.join(mod_path, "stats")
os.makedirs(stats_folder, exist_ok=True)
for l in glob.glob('metrics/stats_fid/*'):
    shutil.copy(l, stats_folder)
    print(f'copy {l} to {stats_folder}')
