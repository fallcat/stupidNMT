import os
import sys
import tarfile
import zipfile
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.request import urlretrieve


DATA_SOURCE = {
    'en_ro': {
        'urls': [
            ('europarl.tgz', 'http://data.statmt.org/wmt16/translation-task/training-parallel-ep-v8.tgz'),
            ('setimes2.zip', 'http://opus.nlpl.eu/download.php?f=SETIMES/v2/moses/en-ro.txt.zip'),
            ('dev.tgz', 'http://data.statmt.org/wmt16/translation-task/dev.tgz'),
            ('test.tgz', 'http://data.statmt.org/wmt16/translation-task/test.tgz')
        ],

        'raw_splits': {
            'train': [
                ('training-parallel-ep-v8/europarl-v8.ro-en.en', 'training-parallel-ep-v8/europarl-v8.ro-en.ro'),
                ('en-ro/SETIMES.en-ro.en', 'en-ro/SETIMES.en-ro.ro')
            ],
            'dev': [
                ('dev/newsdev2016-roen-ref.en.sgm', 'dev/newsdev2016-enro-ref.ro.sgm')
            ],
            'test': [
                ('test/newstest2016-roen-ref.en.sgm', 'test/newstest2016-enro-ref.ro.sgm')
            ]
        }
    },

    'en_ja': {
        'urls': [
            ('iwslt_en_ja.tgz', 'https://wit3.fbk.eu/archive/2017-01-trnted/texts/en/ja/en-ja.tgz'),
            ('iwslt_en_ja_test.tgz',
             'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/en/ja/en-ja.tgz'),
            ('iwslt_en_ja_test_ja_en.tgz', 'https://wit3.fbk.eu/archive/2017-01-ted-test/texts/ja/en/ja-en.tgz')
        ],

        'raw_splits': {
            'train': [
                ('en-ja/train.tags.en-ja.en', 'en-ja/train.tags.en-ja.ja')
            ],
            'dev': [
                ('en-ja/IWSLT17.TED.dev2010.en-ja.en.xml', 'en-ja/IWSLT17.TED.dev2010.en-ja.ja.xml'),
                ('en-ja/IWSLT17.TED.tst2010.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2010.en-ja.ja.xml'),
                ('en-ja/IWSLT17.TED.tst2011.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2011.en-ja.ja.xml'),
                ('en-ja/IWSLT17.TED.tst2012.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2012.en-ja.ja.xml'),
                ('en-ja/IWSLT17.TED.tst2013.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2013.en-ja.ja.xml'),
                ('en-ja/IWSLT17.TED.tst2014.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2014.en-ja.ja.xml'),
                ('en-ja/IWSLT17.TED.tst2015.en-ja.en.xml', 'en-ja/IWSLT17.TED.tst2015.en-ja.ja.xml')
            ],
            'test': [
                ('en-ja/IWSLT17.TED.tst2016.en-ja.en.xml', 'ja-en/IWSLT17.TED.tst2016.ja-en.ja.xml'),
                ('en-ja/IWSLT17.TED.tst2017.en-ja.en.xml', 'ja-en/IWSLT17.TED.tst2017.ja-en.ja.xml')
            ]
        }
    }
}


class DownloadProgressBar(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def __init__(self, filename):
        ''' '''
        super(DownloadProgressBar, self).__init__(
            unit='B', unit_scale=True, miniters=1, desc=filename)

    def update_to(self, blocks=1, block_size=1, total_size=None):
        """
        blocks  : int, optional
            Number of blocks transferred so far [default: 1].
        block_size  : int, optional
            Size of each block (in tqdm units) [default: 1].
        total_size  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if total_size:
            self.total = total_size

        self.update(blocks * block_size - self.n)  # will also set self.n = blocks * block_size


def maybe_download(filepath, url):
    ''' Download the requested URL to the requested path if it does not already exist '''
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(filepath):
        return filepath

    filename = os.path.basename(filepath)
    with DownloadProgressBar(filename) as progress:
        urlretrieve(url, filepath, reporthook=progress.update_to)

    return filepath


def extract_all(filename, path):
    '''
    Extracts all the files in the given archive. Seamlessly determines archive type and compression
    '''
    if tarfile.is_tarfile(filename):
        with tarfile.open(filename, 'r') as archive:
            archive.extractall(path)
    elif zipfile.is_zipfile(filename):
        with zipfile.ZipFile(filename, 'r') as archive:
            archive.extractall(path)
    else:
        print(path)
        print(filename)
        raise ValueError('Unknown archive type!')


def download_and_extract(urls, data_directory):
    ''' Download and extract the dataset '''
    for filename, url in urls:
        filepath = os.path.join(data_directory, filename)
        maybe_download(filepath, url)
        extract_all(filepath, data_directory)


def extract_xml(data_dir):
    pair = ['en', 'ja']
    for split in DATA_SOURCE['en_ja']['raw_splits']:
        for j, data_pair in enumerate(DATA_SOURCE['en_ja']['raw_splits'][split]):
            for i, lang in enumerate(pair):
                raw_path = os.path.join(data_dir, data_pair[i])
                out_path = os.path.join(data_dir, split + '.raw.' + lang)
                if os.path.exists(out_path) and j == 0:
                    os.remove(out_path)
                with open(os.path.join(data_dir, split + '.raw.' + lang), 'at') as output_file:
                    if split == 'train':
                        with open(raw_path, 'rt') as in_file:
                            for line in in_file.readlines():
                                if not line.strip().startswith('<'):
                                    output_file.write(line.strip() + '\n')
                    else:
                        page = open(raw_path)
                        soup = BeautifulSoup(page.read())
                        for line in soup.find_all('seg'):
                            output_file.write(line.string.strip() + '\n')


if __name__ == "__main__":
    lang = sys.argv[1]      # en_ro or en_ja
    data_dir = sys.argv[2]
    download_and_extract(DATA_SOURCE[lang]['urls'], data_dir)
    print(lang)
    if lang == 'en_ja':
        extract_xml(data_dir)
