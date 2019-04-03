import os
import zipfile


def extract(src_dir, dest_dir):
    fails = []
    for file in os.listdir(src_dir):
        name, ext = os.path.splitext(file)
        if ext != '.zip' or name[3:] != '_P':
            continue
        try:
            z = zipfile.ZipFile(src_dir + '/' + file, 'r')
            for zipped_file in z.namelist():
                if zipped_file[3:] == '_TRANSCRIPT.csv':
                    with open(dest_dir+'/'+zipped_file, 'wb') as f:
                        f.write(z.read(zipped_file))
        except zipfile.BadZipFile:
            fails.append(file)
    print('failed for: '+str(fails))


src = os.getenv('DATA_SOURCE_DIR')
dest = './transcripts'
extract(src, dest)
