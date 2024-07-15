import logging
import pandas as pd
import os

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def file_exists(file_path):
    exists = os.path.exists(file_path)
    if not exists:
        logger.debug('File not found: %s' % file_path)
    return exists

class Handler:
    def __init__(self, base_path):
        self.tcia = TciaHandler(base_path)
        logger.info('Using local directory: %s' % base_path)
    

class TciaHandler:
    def __init__(self, tcia_path, img_suffix, excel_file_name):
        self.TCIA_IMG_PATH = tcia_path + 'TCIA_image_PV/'
        self.IMG_SUFFIX = img_suffix
        self.TCIA_RESULTS_PATH = tcia_path + 'TCIA_results_phase_PV/'
        self.TCIA_EXCEL = tcia_path + excel_file_name
        self.df = None

    def __img_join__(self, id):
        return os.path.join(self.TCIA_IMG_PATH, id + self.IMG_SUFFIX)

    def __results_join__(self, id):
        return os.path.join(self.TCIA_RESULTS_PATH, id + self.IMG_SUFFIX)

    def read_df(self):
        logger.debug('reading file...')
        excel_df = pd.read_excel(self.TCIA_EXCEL)
        self.df = excel_df[['TCIA_ID', 'BCLC']]
        logger.info('%d rows in the excel file' % len(self.df))

    def remove_non_target(self):
        before = len(self.df)
        self.df = self.df[self.df['BCLC'] != 'Stage-D']
        logger.info('Removed %d stage-d elements' % (before - len(self.df)))

        
    def classify(self):
        logger.debug('Classifying...')
        mapping = {'Stage-A': 0, 'Stage-B': 1, 'Stage-C': 2}
        map_series = self.df['BCLC'].map(mapping)
        self.df['class'] = map_series
    
    def build_paths(self):
        logger.debug('Looking for paths against contents')
        self.df['img'] = self.df['TCIA_ID'].apply(self.__img_join__)
        self.df['mask'] = self.df['TCIA_ID'].apply(self.__results_join__)

    def remove_not_found(self):
        original = len(self.df)
        self.df = self.df[self.df['img'].apply(file_exists)]
        result = original - len(self.df)
        if result > 0:
            logger.warning('%d files not found' % result)

    def get_result_df(self):
        return self.df[['class', 'img', 'mask']]


if __name__ == '__main__':
    BASE_PATH = '../Data/'
    TCIA_IMG_SUFFIX = '_PV.nii.gz'
    TCIA_LOCATION = BASE_PATH + 'TCIA/'
    TCIA_EXCEL_NAME = 'HCC-TACE-Seg_clinical_data-V2.xlsx'

    logger.debug('Reading tcia...')
    tcia = TciaHandler(TCIA_LOCATION, TCIA_IMG_SUFFIX, TCIA_EXCEL_NAME)
    tcia.read_df()
    tcia.remove_non_target()
    tcia.classify()
    tcia.build_paths()
    tcia.remove_not_found()

    logger.debug(tcia.get_result_df().info())